import { ref } from 'vue'
import { defineStore } from 'pinia'
import { apiFetch } from '@/api/http'

export interface Database {
  db_id: string
  name: string
  description?: string
  embed_model?: string
  dimension?: number
  files?: Record<string, any>
  [key: string]: any
}

export type DatabaseCollection = Record<string, Database> | Database[]

export const useDatabaseStore = defineStore('database', () => {
  const db = ref<DatabaseCollection>({})

  function setDatabase(newDatabase: DatabaseCollection) {
    db.value = newDatabase
  }

  async function refreshDatabase() {
    try {
      const data = await apiFetch<{ databases: DatabaseCollection }>('/data/', { method: 'GET' })
      setDatabase(data?.databases || {})
    } catch (error) {
      console.error('Failed to refresh database store:', error)
      setDatabase({})
    }
  }

  return { db, setDatabase, refreshDatabase }
})
