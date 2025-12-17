# AI 测试代理管理平台 - 前端

基于 React + Vite + TypeScript + Ant Design 构建的前端管理平台。

## 功能特性

- ✅ 任务管理：添加任务、查看任务列表、查看任务详情
- ✅ 业务知识库：完整的 CRUD 操作 + 向量搜索
- ✅ 推理知识库：完整的 CRUD 操作 + 向量搜索
- ✅ 现代化 UI：使用 Ant Design 组件库
- ✅ 响应式设计：适配不同屏幕尺寸

## 技术栈

- **React 18** - UI 框架
- **TypeScript** - 类型安全
- **Vite** - 构建工具
- **Ant Design 5** - UI 组件库
- **React Router** - 路由管理
- **Axios** - HTTP 客户端

## 快速开始

### 安装依赖

```bash
cd frontend
npm install
```

### 启动开发服务器

```bash
npm run dev
```

访问 http://localhost:3000

### 构建生产版本

```bash
npm run build
```

### 预览生产构建

```bash
npm run preview
```

## 项目结构

```
frontend/
├── src/
│   ├── components/      # 公共组件
│   │   └── Layout.tsx   # 布局组件
│   ├── pages/           # 页面组件
│   │   ├── TasksPage.tsx              # 任务管理页面
│   │   ├── BusinessKnowledgePage.tsx   # 业务知识库页面
│   │   └── ReasoningKnowledgePage.tsx # 推理知识库页面
│   ├── services/        # API 服务
│   │   └── api.ts       # API 接口定义
│   ├── App.tsx          # 根组件
│   ├── main.tsx         # 入口文件
│   └── index.css        # 全局样式
├── index.html           # HTML 模板
├── package.json         # 项目配置
├── tsconfig.json        # TypeScript 配置
└── vite.config.ts       # Vite 配置
```

## API 配置

前端通过代理访问后端 API，代理配置在 `vite.config.ts` 中：

```typescript
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    },
  },
}
```

确保后端 API 服务运行在 `http://localhost:8000`。

## 使用说明

### 任务管理

1. 点击"添加任务"按钮
2. 输入任务描述
3. 点击"启动任务"提交
4. 任务会在后台执行，可以通过"查看详情"查看执行结果

### 知识库管理

#### 业务知识库

- **知识列表**：查看、编辑、删除业务知识条目
- **知识搜索**：根据问题或答案进行向量相似度搜索

#### 推理知识库

- **知识列表**：查看、编辑、删除推理知识条目
- **知识搜索**：根据任务或步骤进行向量相似度搜索

## 开发说明

### 添加新页面

1. 在 `src/pages/` 目录下创建新页面组件
2. 在 `src/App.tsx` 中添加路由
3. 在 `src/components/Layout.tsx` 中添加菜单项

### 添加新 API

在 `src/services/api.ts` 中添加新的 API 方法。

## 注意事项

- 确保后端 API 服务已启动并运行在 `http://localhost:8000`
- 如果后端运行在不同端口，需要修改 `vite.config.ts` 中的代理配置

