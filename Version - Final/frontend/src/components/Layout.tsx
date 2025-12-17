import { useState } from 'react'
import { Layout, Menu, theme } from 'antd'
import { useNavigate, useLocation } from 'react-router-dom'
import {
  PlayCircleOutlined,
  DatabaseOutlined,
  BookOutlined,
} from '@ant-design/icons'

const { Header, Content, Sider } = Layout

interface AppLayoutProps {
  children: React.ReactNode
}

const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const [collapsed, setCollapsed] = useState(false)
  const navigate = useNavigate()
  const location = useLocation()
  const {
    token: { colorBgContainer },
  } = theme.useToken()

  const menuItems = [
    {
      key: '/tasks',
      icon: <PlayCircleOutlined />,
      label: '任务管理',
    },
    {
      key: '/business-knowledge',
      icon: <DatabaseOutlined />,
      label: '业务知识库',
    },
    {
      key: '/reasoning-knowledge',
      icon: <BookOutlined />,
      label: '推理知识库',
    },
  ]

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider collapsible collapsed={collapsed} onCollapse={setCollapsed}>
        <div
          style={{
            height: 32,
            margin: 16,
            background: 'rgba(255, 255, 255, 0.3)',
            borderRadius: 6,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'white',
            fontWeight: 'bold',
          }}
        >
          {collapsed ? 'AI' : 'AI 测试智能体'}
        </div>
        <Menu
          theme="dark"
          selectedKeys={[location.pathname]}
          mode="inline"
          items={menuItems}
          onClick={({ key }) => navigate(key)}
        />
      </Sider>
      <Layout>
        <Header style={{ padding: 0, background: colorBgContainer }}>
          <div style={{ padding: '0 24px', fontSize: 18, fontWeight: 'bold' }}>
            AI 自动化测试管理平台
          </div>
        </Header>
        <Content style={{ margin: '24px 16px', padding: 24, background: colorBgContainer }}>
          {children}
        </Content>
      </Layout>
    </Layout>
  )
}

export default AppLayout

