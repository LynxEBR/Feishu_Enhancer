import { useState, useEffect } from 'react'
import {
  Table,
  Button,
  Input,
  Space,
  Modal,
  message,
  Tag,
  Descriptions,
  Card,
  Typography,
} from 'antd'
import { PlusOutlined, SearchOutlined, DeleteOutlined } from '@ant-design/icons'
import { taskApi } from '../services/api'
import type { ColumnsType } from 'antd/es/table'
import dayjs from 'dayjs'

const { TextArea } = Input
const { Title, Text } = Typography

interface Task {
  id: number
  original_task: string
  enhanced_task?: string
  can_execute?: boolean
  execution_reason?: string
  steps?: string
  step_results?: any
  final_result?: string
  all_success?: boolean
}

const TasksPage = () => {
  const [tasks, setTasks] = useState<Task[]>([])
  const [loading, setLoading] = useState(false)
  const [modalVisible, setModalVisible] = useState(false)
  const [detailModalVisible, setDetailModalVisible] = useState(false)
  const [selectedTask, setSelectedTask] = useState<Task | null>(null)
  const [taskInput, setTaskInput] = useState('')

  const fetchTasks = async () => {
    setLoading(true)
    try {
      const response = await taskApi.getTasks(0, 100)
      setTasks(response.data)
    } catch (error: any) {
      message.error('获取任务列表失败: ' + (error.response?.data?.detail || error.message))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchTasks()
  }, [])

  const handleRunTask = async () => {
    if (!taskInput.trim()) {
      message.warning('请输入任务描述')
      return
    }

    try {
      const response = await taskApi.runTask(taskInput.trim())
      message.success('任务已添加到执行队列，ID: ' + response.data.task_id)
      setModalVisible(false)
      setTaskInput('')
      // 等待一下再刷新列表，让任务记录有时间创建
      setTimeout(() => {
        fetchTasks()
      }, 1000)
    } catch (error: any) {
      message.error('启动任务失败: ' + (error.response?.data?.detail || error.message))
    }
  }

  const handleViewDetail = async (taskId: number) => {
    try {
      const response = await taskApi.getTask(taskId)
      setSelectedTask(response.data)
      setDetailModalVisible(true)
    } catch (error: any) {
      message.error('获取任务详情失败: ' + (error.response?.data?.detail || error.message))
    }
  }

  const handleDelete = async (taskId: number) => {
    Modal.confirm({
      title: '确认删除',
      content: '确定要删除这个任务吗？',
      onOk: async () => {
        try {
          await taskApi.deleteTask(taskId)
          message.success('删除成功')
          fetchTasks()
        } catch (error: any) {
          message.error('删除失败: ' + (error.response?.data?.detail || error.message))
        }
      },
    })
  }

  const columns: ColumnsType<Task> = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 80,
    },
    {
      title: '原始任务',
      dataIndex: 'original_task',
      key: 'original_task',
      ellipsis: true,
      render: (text: string) => text || '-',
    },
    {
      title: '增强任务',
      dataIndex: 'enhanced_task',
      key: 'enhanced_task',
      ellipsis: true,
      render: (text: string) => text || '-',
    },
    {
      title: '是否执行成功',
      dataIndex: 'all_success',
      key: 'all_success',
      width: 120,
      render: (allSuccess: boolean) =>
        allSuccess === null || allSuccess === undefined ? (
          <Tag color="default">未知</Tag>
        ) : allSuccess ? (
          <Tag color="success">是</Tag>
        ) : (
          <Tag color="error">否</Tag>
        ),
    },
    {
      title: '最终结果',
      dataIndex: 'final_result',
      key: 'final_result',
      ellipsis: true,
      width: 200,
      render: (text: string) => {
        if (!text) return '-'
        try {
          // 尝试解析 JSON 格式的最终结果
          const parsed = JSON.parse(text)
          return parsed.summary || text.substring(0, 50) + '...'
        } catch {
          // 如果不是 JSON，直接显示文本（截断）
          return text.length > 50 ? text.substring(0, 50) + '...' : text
        }
      },
    },
    {
      title: '判断是否可执行',
      dataIndex: 'can_execute',
      key: 'can_execute',
      width: 100,
      render: (canExecute: boolean) =>
        canExecute === null || canExecute === undefined ? (
          <Tag color="default">未知</Tag>
        ) : canExecute ? (
          <Tag color="success">是</Tag>
        ) : (
          <Tag color="error">否</Tag>
        ),
    },
    {
      title: '操作',
      key: 'action',
      width: 200,
      render: (_, record) => (
        <Space>
          <Button type="link" size="small" onClick={() => handleViewDetail(record.id)}>
            查看详情
          </Button>
          <Button
            type="link"
            size="small"
            danger
            onClick={() => handleDelete(record.id)}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ]

  return (
    <div>
      <Space style={{ marginBottom: 16 }}>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalVisible(true)}>
          添加任务
        </Button>
        <Button icon={<SearchOutlined />} onClick={fetchTasks}>
          刷新
        </Button>
      </Space>

      <Table
        columns={columns}
        dataSource={tasks}
        rowKey="id"
        loading={loading}
        pagination={{ pageSize: 10 }}
      />

      <Modal
        title="添加任务"
        open={modalVisible}
        onOk={handleRunTask}
        onCancel={() => {
          setModalVisible(false)
          setTaskInput('')
        }}
        okText="启动任务"
        cancelText="取消"
        width={600}
      >
        <Space direction="vertical" style={{ width: '100%' }} size="large">
          <div>
            <Text strong>任务描述：</Text>
            <TextArea
              rows={4}
              value={taskInput}
              onChange={(e) => setTaskInput(e.target.value)}
              placeholder="请输入任务描述，例如：云文档被分享到IM中能正常打开"
            />
          </div>
        </Space>
      </Modal>

      <Modal
        title="任务详情"
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        footer={null}
        width={800}
      >
        {selectedTask && (
          <Descriptions column={1} bordered>
            <Descriptions.Item label="ID">{selectedTask.id}</Descriptions.Item>
            <Descriptions.Item label="原始任务">
              {selectedTask.original_task || '-'}
            </Descriptions.Item>
            <Descriptions.Item label="增强任务">
              {selectedTask.enhanced_task || '-'}
            </Descriptions.Item>
            <Descriptions.Item label="全部成功">
              {selectedTask.all_success === null || selectedTask.all_success === undefined
                ? '未知'
                : selectedTask.all_success
                ? '是'
                : '否'}
            </Descriptions.Item>
            <Descriptions.Item label="可执行">
              {selectedTask.can_execute === null || selectedTask.can_execute === undefined
                ? '未知'
                : selectedTask.can_execute
                ? '是'
                : '否'}
            </Descriptions.Item>
            <Descriptions.Item label="执行原因">
              {selectedTask.execution_reason || '-'}
            </Descriptions.Item>
            <Descriptions.Item label="步骤">
              <pre style={{ whiteSpace: 'pre-wrap', maxHeight: 200, overflow: 'auto' }}>
                {selectedTask.steps || '-'}
              </pre>
            </Descriptions.Item>
            <Descriptions.Item label="步骤结果">
              <pre style={{ whiteSpace: 'pre-wrap', maxHeight: 200, overflow: 'auto' }}>
                {JSON.stringify(selectedTask.step_results || [], null, 2)}
              </pre>
            </Descriptions.Item>
            <Descriptions.Item label="最终结果">
              <pre style={{ whiteSpace: 'pre-wrap', maxHeight: 200, overflow: 'auto' }}>
                {selectedTask.final_result || '-'}
              </pre>
            </Descriptions.Item>
          </Descriptions>
        )}
      </Modal>
    </div>
  )
}

export default TasksPage

