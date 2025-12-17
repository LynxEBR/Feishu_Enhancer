import { useState, useEffect } from 'react'
import {
  Table,
  Button,
  Input,
  Space,
  Modal,
  message,
  Form,
  Popconfirm,
  Card,
  Tabs,
  InputNumber,
} from 'antd'
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  SearchOutlined,
} from '@ant-design/icons'
import { reasoningKnowledgeApi } from '../services/api'
import type { ColumnsType } from 'antd/es/table'

const { TextArea } = Input
const { TabPane } = Tabs

interface ReasoningKnowledge {
  id: number
  task_text: string
  step_text: string
  created_at?: string
  updated_at?: string
  similarity?: number
}

const ReasoningKnowledgePage = () => {
  const [knowledgeList, setKnowledgeList] = useState<ReasoningKnowledge[]>([])
  const [loading, setLoading] = useState(false)
  const [modalVisible, setModalVisible] = useState(false)
  const [editingItem, setEditingItem] = useState<ReasoningKnowledge | null>(null)
  const [form] = Form.useForm()
  const [searchForm] = Form.useForm()
  const [searchResults, setSearchResults] = useState<ReasoningKnowledge[]>([])
  const [searchLoading, setSearchLoading] = useState(false)

  const fetchKnowledge = async () => {
    setLoading(true)
    try {
      const response = await reasoningKnowledgeApi.list(0, 100)
      setKnowledgeList(response.data)
    } catch (error: any) {
      message.error('获取知识列表失败: ' + (error.response?.data?.detail || error.message))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchKnowledge()
  }, [])

  const handleCreate = () => {
    setEditingItem(null)
    form.resetFields()
    setModalVisible(true)
  }

  const handleEdit = (record: ReasoningKnowledge) => {
    setEditingItem(record)
    form.setFieldsValue({
      task_text: record.task_text,
      step_text: record.step_text,
    })
    setModalVisible(true)
  }

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields()
      if (editingItem) {
        await reasoningKnowledgeApi.update(editingItem.id, values)
        message.success('更新成功')
      } else {
        await reasoningKnowledgeApi.create(values)
        message.success('创建成功')
      }
      setModalVisible(false)
      fetchKnowledge()
    } catch (error: any) {
      if (error.errorFields) {
        return
      }
      message.error('操作失败: ' + (error.response?.data?.detail || error.message))
    }
  }

  const handleDelete = async (id: number) => {
    try {
      await reasoningKnowledgeApi.delete(id)
      message.success('删除成功')
      fetchKnowledge()
    } catch (error: any) {
      message.error('删除失败: ' + (error.response?.data?.detail || error.message))
    }
  }

  const handleSearch = async (type: 'task' | 'step') => {
    try {
      const values = await searchForm.validateFields()
      setSearchLoading(true)
      const response =
        type === 'task'
          ? await reasoningKnowledgeApi.searchByTask(
              values.query_text,
              values.top_k || 5,
              values.threshold || 0.0,
            )
          : await reasoningKnowledgeApi.searchByStep(
              values.query_text,
              values.top_k || 5,
              values.threshold || 0.0,
            )
      setSearchResults(response.data)
    } catch (error: any) {
      if (error.errorFields) {
        return
      }
      message.error('搜索失败: ' + (error.response?.data?.detail || error.message))
    } finally {
      setSearchLoading(false)
    }
  }

  const columns: ColumnsType<ReasoningKnowledge> = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 80,
    },
    {
      title: '任务',
      dataIndex: 'task_text',
      key: 'task_text',
      ellipsis: true,
    },
    {
      title: '步骤',
      dataIndex: 'step_text',
      key: 'step_text',
      ellipsis: true,
    },
    {
      title: '相似度',
      dataIndex: 'similarity',
      key: 'similarity',
      width: 100,
      render: (similarity: number) =>
        similarity !== undefined ? (similarity * 100).toFixed(2) + '%' : '-',
    },
    {
      title: '操作',
      key: 'action',
      width: 150,
      render: (_, record) => (
        <Space>
          <Button type="link" size="small" icon={<EditOutlined />} onClick={() => handleEdit(record)}>
            编辑
          </Button>
          <Popconfirm
            title="确定要删除吗？"
            onConfirm={() => handleDelete(record.id)}
            okText="确定"
            cancelText="取消"
          >
            <Button type="link" size="small" danger icon={<DeleteOutlined />}>
              删除
            </Button>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <div>
      <Tabs defaultActiveKey="list">
        <TabPane tab="知识列表" key="list">
          <Space style={{ marginBottom: 16 }}>
            <Button type="primary" icon={<PlusOutlined />} onClick={handleCreate}>
              添加知识
            </Button>
            <Button icon={<SearchOutlined />} onClick={fetchKnowledge}>
              刷新
            </Button>
          </Space>

          <Table
            columns={columns}
            dataSource={knowledgeList}
            rowKey="id"
            loading={loading}
            pagination={{ pageSize: 10 }}
          />
        </TabPane>

        <TabPane tab="知识搜索" key="search">
          <Card>
            <Form form={searchForm} layout="vertical">
              <Form.Item
                name="query_text"
                label="查询文本"
                rules={[{ required: true, message: '请输入查询文本' }]}
              >
                <TextArea rows={3} placeholder="输入要搜索的任务或步骤" />
              </Form.Item>
              <Form.Item name="top_k" label="返回数量" initialValue={5}>
                <InputNumber min={1} max={50} />
              </Form.Item>
              <Form.Item name="threshold" label="相似度阈值" initialValue={0.0}>
                <InputNumber min={0} max={1} step={0.1} />
              </Form.Item>
              <Space>
                <Button
                  type="primary"
                  onClick={() => handleSearch('task')}
                  loading={searchLoading}
                >
                  按任务搜索
                </Button>
                <Button
                  type="primary"
                  onClick={() => handleSearch('step')}
                  loading={searchLoading}
                >
                  按步骤搜索
                </Button>
              </Space>
            </Form>
          </Card>

          <Table
            columns={columns}
            dataSource={searchResults}
            rowKey="id"
            loading={searchLoading}
            style={{ marginTop: 16 }}
            pagination={{ pageSize: 10 }}
          />
        </TabPane>
      </Tabs>

      <Modal
        title={editingItem ? '编辑知识' : '添加知识'}
        open={modalVisible}
        onOk={handleSubmit}
        onCancel={() => {
          setModalVisible(false)
          form.resetFields()
        }}
        okText="确定"
        cancelText="取消"
        width={600}
      >
        <Form form={form} layout="vertical">
          <Form.Item
            name="task_text"
            label="任务"
            rules={[{ required: true, message: '请输入任务' }]}
          >
            <TextArea rows={3} placeholder="请输入任务描述" />
          </Form.Item>
          <Form.Item
            name="step_text"
            label="步骤"
            rules={[{ required: true, message: '请输入步骤' }]}
          >
            <TextArea rows={5} placeholder="请输入步骤描述" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default ReasoningKnowledgePage

