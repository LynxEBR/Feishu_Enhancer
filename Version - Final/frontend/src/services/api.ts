import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

// 任务相关 API
export const taskApi = {
  // 启动任务
  runTask: (task: string, taskId?: number) =>
    api.post('/tasks/run', { task, task_id: taskId }),

  // 获取任务详情
  getTask: (taskId: number) => api.get(`/tasks/${taskId}`),

  // 获取任务列表
  getTasks: (skip = 0, limit = 100) =>
    api.get('/tasks', { params: { skip, limit } }),

  // 删除任务
  deleteTask: (taskId: number) => api.delete(`/tasks/${taskId}`),
}

// 业务知识库 API
export const businessKnowledgeApi = {
  // 创建
  create: (data: { question_text: string; answer_text: string }) =>
    api.post('/business-knowledge', data),

  // 获取详情
  get: (id: number) => api.get(`/business-knowledge/${id}`),

  // 获取列表
  list: (skip = 0, limit = 100) =>
    api.get('/business-knowledge', { params: { skip, limit } }),

  // 更新
  update: (id: number, data: { question_text?: string; answer_text?: string }) =>
    api.put(`/business-knowledge/${id}`, data),

  // 删除
  delete: (id: number) => api.delete(`/business-knowledge/${id}`),

  // 根据问题搜索
  searchByQuestion: (query: string, topK = 5, threshold = 0.0) =>
    api.post('/business-knowledge/search/question', {
      query_text: query,
      top_k: topK,
      threshold,
    }),

  // 根据答案搜索
  searchByAnswer: (query: string, topK = 5, threshold = 0.0) =>
    api.post('/business-knowledge/search/answer', {
      query_text: query,
      top_k: topK,
      threshold,
    }),

  // 获取总数
  count: () => api.get('/business-knowledge/count'),
}

// 推理知识库 API
export const reasoningKnowledgeApi = {
  // 创建
  create: (data: { task_text: string; step_text: string }) =>
    api.post('/reasoning-knowledge', data),

  // 获取详情
  get: (id: number) => api.get(`/reasoning-knowledge/${id}`),

  // 获取列表
  list: (skip = 0, limit = 100) =>
    api.get('/reasoning-knowledge', { params: { skip, limit } }),

  // 更新
  update: (id: number, data: { task_text?: string; step_text?: string }) =>
    api.put(`/reasoning-knowledge/${id}`, data),

  // 删除
  delete: (id: number) => api.delete(`/reasoning-knowledge/${id}`),

  // 根据任务搜索
  searchByTask: (query: string, topK = 5, threshold = 0.0) =>
    api.post('/reasoning-knowledge/search/task', {
      query_text: query,
      top_k: topK,
      threshold,
    }),

  // 根据步骤搜索
  searchByStep: (query: string, topK = 5, threshold = 0.0) =>
    api.post('/reasoning-knowledge/search/step', {
      query_text: query,
      top_k: topK,
      threshold,
    }),

  // 获取总数
  count: () => api.get('/reasoning-knowledge/count'),
}

export default api

