import { ref, computed } from 'vue'
import { defineStore } from 'pinia'
import { apiFetch } from '@/api/http'

export const useDatabaseStore = defineStore('database', () => {
  const db = ref({})
  function setDatabase(newDatabase) {
    db.value = newDatabase
  }

  function refreshDatabase() {
    apiFetch('/api/data/', { method: 'GET' })
      .then((data) => {
        setDatabase(data?.databases || {})
      })
      .catch(() => {
        setDatabase({})
      })
  }

  return { db, setDatabase, refreshDatabase }
})
