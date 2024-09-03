/*
 * Copyright © 2022 Collabora, Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

#include <sys/stat.h>

#include "detect_os.h"
#include "string.h"
#include "mesa_cache_db_multipart.h"
#include "u_debug.h"

bool
mesa_cache_db_multipart_open(struct mesa_cache_db_multipart *db,
                             const char *cache_path)
{
#if DETECT_OS_WINDOWS
   return false;
#else
   char *part_path = NULL;
   unsigned int i;

   db->num_parts = debug_get_num_option("MESA_DISK_CACHE_DATABASE_NUM_PARTS", 10);
   db->cache_path = cache_path;
   db->parts = calloc(db->num_parts, sizeof(*db->parts));
   if (!db->parts)
      return false;

   mtx_init(&db->lock, mtx_plain);

   return true;
#endif
}

static bool
mesa_cache_db_multipart_open_part_locked(struct mesa_cache_db_multipart *db,
                                         unsigned int part)
{
#if DETECT_OS_WINDOWS
   return false;
#else
   bool db_opened = false;
   char *part_path = NULL;

   if (db->parts[part])
      return true;

   if (asprintf(&part_path, "%s/part%u", db->cache_path, part) == -1)
      return false;

   if (mkdir(part_path, 0755) == -1 && errno != EEXIST)
      goto free_path;

   db->parts[part] = calloc(1, sizeof(*db->parts[part]));

   /* DB opening may fail only in a case of a severe problem,
    * like IO error.
    */
   db_opened = mesa_cache_db_open(db->parts[part], part_path);
   if (!db_opened) {
      free(db->parts[part]);
      db->parts[part] = NULL;
      goto free_path;
   }

   if (db->max_cache_size)
      mesa_cache_db_set_size_limit(db->parts[part],
                                   db->max_cache_size / db->num_parts);

   /* remove old pre multi-part cache */
   mesa_db_wipe_path(db->cache_path);

free_path:
   free(part_path);

   return db_opened;
#endif
}

void
mesa_cache_db_multipart_close(struct mesa_cache_db_multipart *db)
{
   while (db->num_parts--) {
      if (db->parts[db->num_parts]) {
         mesa_cache_db_close(db->parts[db->num_parts]);
         free(db->parts[db->num_parts]);
      }
   }

   free(db->parts);
   mtx_destroy(&db->lock);
}

void
mesa_cache_db_multipart_set_size_limit(struct mesa_cache_db_multipart *db,
                                       uint64_t max_cache_size)
{
   for (unsigned int part = 0; part < db->num_parts; part++) {
      if (db->parts[part])
         mesa_cache_db_set_size_limit(db->parts[part],
                                      max_cache_size / db->num_parts);
   }

   db->max_cache_size = max_cache_size;
}

static void *
mesa_cache_db_multipart_read_entry_locked(struct mesa_cache_db_multipart *db,
                                          const uint8_t *cache_key_160bit,
                                          size_t *size)
{
   unsigned last_read_part = db->last_read_part;

   for (unsigned int i = 0; i < db->num_parts; i++) {
      unsigned int part = (last_read_part + i) % db->num_parts;

      if (!mesa_cache_db_multipart_open_part_locked(db, part))
         break;

      void *cache_item = mesa_cache_db_read_entry(db->parts[part],
                                                  cache_key_160bit, size);
      if (cache_item) {
         /* Likely that the next entry lookup will hit the same DB part. */
         db->last_read_part = part;
         return cache_item;
      }
   }

   return NULL;
}

void *
mesa_cache_db_multipart_read_entry(struct mesa_cache_db_multipart *db,
                                   const uint8_t *cache_key_160bit,
                                   size_t *size)
{
   void *cache_item;

   mtx_lock(&db->lock);
   cache_item = mesa_cache_db_multipart_read_entry_locked(db, cache_key_160bit,
                                                          size);
   mtx_unlock(&db->lock);

   return cache_item;
}

static unsigned
mesa_cache_db_multipart_select_victim_part(struct mesa_cache_db_multipart *db)
{
   double best_score = 0, score;
   unsigned victim = 0;

   for (unsigned int i = 0; i < db->num_parts; i++) {
      if (!mesa_cache_db_multipart_open_part_locked(db, i))
         continue;

      score = mesa_cache_db_eviction_score(db->parts[i]);
      if (score > best_score) {
         best_score = score;
         victim = i;
      }
   }

   return victim;
}

static bool
mesa_cache_db_multipart_entry_write_locked(struct mesa_cache_db_multipart *db,
                                           const uint8_t *cache_key_160bit,
                                           const void *blob, size_t blob_size)
{
   unsigned last_written_part = db->last_written_part;
   int wpart = -1;

   for (unsigned int i = 0; i < db->num_parts; i++) {
      unsigned int part = (last_written_part + i) % db->num_parts;

      if (!mesa_cache_db_multipart_open_part_locked(db, part))
         break;

      /* Note that each DB part has own locking. */
      if (mesa_cache_db_has_space(db->parts[part], blob_size)) {
         wpart = part;
         break;
      }
   }

   /* All DB parts are full. Writing to a full DB part will auto-trigger
    * eviction of LRU cache entries from the part. Select DB part that
    * contains majority of LRU cache entries.
    */
   if (wpart < 0)
      wpart = mesa_cache_db_multipart_select_victim_part(db);

   if (!mesa_cache_db_multipart_open_part_locked(db, wpart))
      return false;

   db->last_written_part = wpart;

   return mesa_cache_db_entry_write(db->parts[wpart], cache_key_160bit,
                                    blob, blob_size);
}

bool
mesa_cache_db_multipart_entry_write(struct mesa_cache_db_multipart *db,
                                    const uint8_t *cache_key_160bit,
                                    const void *blob, size_t blob_size)
{
   bool ret;

   mtx_lock(&db->lock);
   ret =  mesa_cache_db_multipart_entry_write_locked(db, cache_key_160bit,
                                                     blob, blob_size);
   mtx_unlock(&db->lock);

   return ret;
}

void
mesa_cache_db_multipart_entry_remove(struct mesa_cache_db_multipart *db,
                                     const uint8_t *cache_key_160bit)
{
   mtx_lock(&db->lock);

   for (unsigned int i = 0; i < db->num_parts; i++) {
      if (!mesa_cache_db_multipart_open_part_locked(db, i))
         continue;

      mesa_cache_db_entry_remove(db->parts[i], cache_key_160bit);
   }

   mtx_unlock(&db->lock);
}
