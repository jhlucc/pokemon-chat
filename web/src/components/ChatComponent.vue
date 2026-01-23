<template>
  <div class="chat"  ref="chatContainer">
<!--     顶部左侧是打开侧边栏、新建会话、切换模型，右侧是打开选项设置面板。>-->
    <div class="chat-header">
<!--      聊天界面顶部导航栏（新建对话、切换模型、选项面板）-->
      <div class="header__left">
        <div
          v-if="!state.isSidebarOpen"
          class="close nav-btn"
          @click="state.isSidebarOpen = true"
        >
          <img src="@/assets/icons/sidebar_left.svg" class="iconfont icon-20" alt="设置" />
        </div>

         <div class="action-button" @click="$emit('newconv')">
        <PlusCircleOutlined class="icon" />
        <span class="text">新建会话</span>
      </div>


      </div>
      <div class="header__right">
         <a-dropdown>
  <div class="model-select" @click.prevent>
    <BulbOutlined class="icon" />
    <span class="text">{{ configStore.config?.model_provider }}/{{ configStore.config?.model_name }}</span>
  </div>
  <template #overlay>
    <a-menu class="scrollable-menu">
      <a-menu-item-group
        v-for="(item, key) in modelKeys"
        :key="key"
        :title="modelNames[item]?.name"
      >
        <a-menu-item
          v-for="(model, idx) in modelNames[item]?.models"
          :key="`${item}-${idx}`"
          @click="selectModel(item, model)"
        >
          {{ item }}/{{ model }}
        </a-menu-item>
      </a-menu-item-group>
      <a-menu-item-group v-if="opts.agents && opts.agents.length > 0" title="智能体 (Agents)">
         <a-menu-item
          v-for="(agent, idx) in opts.agents"
          :key="`agent-${idx}`"
          @click="selectModel('agent', agent.name)"
        >
          agent/{{ agent.name }}
        </a-menu-item>
      </a-menu-item-group>
      <a-menu-item-group v-if="customModels.length > 0" title="自定义模型">
        <a-menu-item
          v-for="(model, idx) in customModels"
          :key="`custom-${idx}`"
          @click="selectModel('custom', model.custom_id)"
        >
          custom/{{ model.custom_id }}
        </a-menu-item>
      </a-menu-item-group>
    </a-menu>
  </template>
</a-dropdown>
        <div class="nav-btn text" @click="opts.showPanel = !opts.showPanel">
          <component :is="opts.showPanel ? FolderOpenOutlined : FolderOutlined" /> <span class="text">选项</span>
        </div>
        <div v-if="opts.showPanel" class="my-panal r0 top100 swing-in-top-fwd" ref="panel">
          <div class="flex-center" @click="meta.stream = !meta.stream">
            流式输出 <div @click.stop><a-switch v-model:checked="meta.stream" /></div>
          </div>
          <div class="flex-center" @click="meta.summary_title = !meta.summary_title">
            总结对话标题 <div @click.stop><a-switch v-model:checked="meta.summary_title" /></div>
          </div>
          <div class="flex-center">
            最大历史轮数 <a-input-number id="inputNumber" v-model:value="meta.history_round" :min="1" :max="50" />
          </div>
          <div class="flex-center">
            字体大小
            <a-select v-model:value="meta.fontSize" style="width: 100px" placeholder="选择字体大小">
              <a-select-option value="smaller">更小</a-select-option>
              <a-select-option value="default">默认</a-select-option>
              <a-select-option value="larger">更大</a-select-option>
            </a-select>
          </div>
          <div class="flex-center" @click="meta.wideScreen = !meta.wideScreen">
            宽屏模式 <div @click.stop><a-switch v-model:checked="meta.wideScreen" /></div>
          </div>

        </div>
      </div>
    </div>

    <!-- Empty State: Centered Greeting & Input -->
    <!-- Empty State: Centered Greeting & Input -->
    <div v-if="conv.messages.length === 0" class="welcome-container">
       <div class="logo-container">
          <img src="@/assets/logo.svg" class="welcome-logo" alt="Logo" />
       </div>
       <div class="greeting">您今天在想什么？</div>
       
       <div class="center-input-wrapper">
        <MessageInputComponent
          v-model="conv.inputText"
          :is-loading="isStreaming"
          :send-button-disabled="!conv.inputText && !isStreaming"
          :auto-size="{ minRows: 1, maxRows: 8 }"
          @send="handleSendOrStop"
          @keydown="handleKeyDown"
          class="center-input"
        >
          <template #options-left>
             <!-- Reuse options logic -->
             <div :class="{'switch': true, 'opt-item': true, 'active': meta.use_web}" v-if="configStore.config.enable_web_search" @click="meta.use_web=!meta.use_web"><CompassOutlined style="margin-right: 3px;"/>联网</div>
             <div :class="{'switch': true, 'opt-item': true, 'active': meta.use_graph}" v-if="configStore.config.enable_knowledge_graph" @click="meta.use_graph=!meta.use_graph"><DeploymentUnitOutlined style="margin-right: 3px;"/>图谱</div>
              <a-dropdown v-if="configStore.config.enable_knowledge_base && opts.databases.length > 0">
                 <div :class="{'opt-item': true, 'active': meta.selectedKB !== null}">
                    <BookOutlined style="margin-right: 3px;"/>{{ meta.selectedKB === null ? '知识库' : opts.databases[meta.selectedKB]?.name }}
                 </div>
                 <template #overlay>
                    <a-menu>
                      <a-menu-item v-for="(db, index) in opts.databases" :key="index" @click="useDatabase(index)">{{ db.name }}</a-menu-item>
                      <a-menu-item @click="useDatabase(null)">不使用</a-menu-item>
                    </a-menu>
                 </template>
              </a-dropdown>
          </template>
        </MessageInputComponent>
        </div>

        <div class="suggestion-chips">
           <div class="chip" @click="conv.inputText = '制定一个旅行计划'"><CompassOutlined /> 制定旅行计划</div>
           <div class="chip" @click="conv.inputText = '帮我写一段Python代码'"><DeploymentUnitOutlined /> 代码助手</div>
           <div class="chip" @click="conv.inputText = '分析一下这张图片'"><FolderOpenOutlined /> 图片分析</div>
           <div class="chip" @click="conv.inputText = '今天有什么新闻？'"><BulbOutlined /> 新闻摘要</div>
        </div>
     </div>

    <!-- Active Chat State -->
    <div v-else class="chat-main-content">
        <div class="chat-box" :class="{ 'wide-screen': meta.wideScreen, 'font-smaller': meta.fontSize === 'smaller', 'font-larger': meta.fontSize === 'larger' }">
          <MessageComponent
            v-for="message in conv.messages"
            :message="message"
            :key="message.id"
            :is-processing="isStreaming"
            :show-refs="true"
            @retry="retryMessage(message.id)"
            @retryStoppedMessage="retryStoppedMessage(message.id)"
          >
          </MessageComponent>
        </div>
        <div class="bottom">
          <div class="message-input-wrapper"  :class="{ 'wide-screen': meta.wideScreen}">
            <MessageInputComponent
              v-model="conv.inputText"
              :is-loading="isStreaming"
              :send-button-disabled="!conv.inputText && !isStreaming"
              :auto-size="{ minRows: 1, maxRows: 10 }"
              @send="handleSendOrStop"
              @keydown="handleKeyDown"
            >
              <template #options-left>
                <div :class="{'switch': true, 'opt-item': true, 'active': meta.use_web}" v-if="configStore.config.enable_web_search" @click="meta.use_web=!meta.use_web"><CompassOutlined style="margin-right: 3px;"/>联网</div>
                <div :class="{'switch': true, 'opt-item': true, 'active': meta.use_graph}" v-if="configStore.config.enable_knowledge_graph" @click="meta.use_graph=!meta.use_graph"><DeploymentUnitOutlined style="margin-right: 3px;"/>图谱</div>
                 <div
                  :class="{'switch': true, 'opt-item': true, 'active': meta.use_mcp}"
                  v-if="configStore.config.enable_mcp"
                  @click="meta.use_mcp = !meta.use_mcp; meta.mcp_id  = meta.use_mcp ? 'default' : null;">
                  <DatabaseOutlined style="margin-right:3px;" />MCP
                </div>
                <a-dropdown v-if="configStore.config.enable_knowledge_base && opts.databases.length > 0">
                 <div :class="{'opt-item': true, 'active': meta.selectedKB !== null}">
                      <BookOutlined style="margin-right: 3px;"/>{{ meta.selectedKB === null ? '知识库' : opts.databases[meta.selectedKB]?.name }}
                 </div>
                  <template #overlay>
                    <a-menu>
                      <a-menu-item v-for="(db, index) in opts.databases" :key="index" @click="useDatabase(index)">{{ db.name }}</a-menu-item>
                      <a-menu-item @click="useDatabase(null)">不使用</a-menu-item>
                    </a-menu>
                  </template>
                </a-dropdown>
              </template>
            </MessageInputComponent>
            <p class="note">请注意辨别内容的可靠性 By {{ configStore.config?.model_provider }}: {{ configStore.config?.model_name }}</p>
          </div>
        </div>
    </div>
  </div>
</template>

<script setup>
import { reactive, ref, onMounted, toRefs, nextTick, onUnmounted, watch, computed } from 'vue'
import {

  BookOutlined,
  CompassOutlined,
  PlusCircleOutlined,
  FolderOutlined,
  FolderOpenOutlined,
  BulbOutlined,
  DeploymentUnitOutlined,
    DatabaseOutlined,

} from '@ant-design/icons-vue'
import { onClickOutside } from '@vueuse/core'
import { useConfigStore } from '@/stores/config'
import { message } from 'ant-design-vue'
import MessageInputComponent from '@/components/MessageInputComponent.vue'
import MessageComponent from '@/components/MessageComponent.vue'

const props = defineProps({
  conv: Object,
  state: Object
})

const emit = defineEmits(['rename-title', 'newconv']);
const configStore = useConfigStore()

const { conv, state } = toRefs(props)
const chatContainer = ref(null)

const isStreaming = ref(false)
const userIsScrolling = ref(false);
const shouldAutoScroll = ref(true);

const panel = ref(null)
const modelCard = ref(null)
const examples = ref([
  '喜欢小智吗？',
  '今天常州天气怎么样？',
  '介绍一下皮卡丘',
  '今天星期几？'
])

const opts = reactive({
  showPanel: false,
  showModelCard: false,
  openDetail: false,
  databases: [],
  mcps: [],
  agents: []
})

const meta = reactive(JSON.parse(localStorage.getItem('meta')) || {
  use_graph: false,
  use_web: false,
  use_mcp: false,
  graph_name: "neo4j",
  selectedKB: null,
  mcp_id: null,
  stream: true,
  summary_title: false,
  history_round: 20,
  db_id: null,
  fontSize: 'default',

  wideScreen: false,
  themeMode: false    // 控制亮/暗色模式
})


const consoleMsg = (msg) => console.log(msg)
onClickOutside(panel, () => setTimeout(() => opts.showPanel = false, 30))
onClickOutside(modelCard, () => setTimeout(() => opts.showModelCard = false, 30))

// 从 message 中获取 history 信息，每个消息都是 {role, content} 的格式
const getHistory = () => {
  const history = conv.value.messages.map((msg) => {
    if (msg.content) {
      return {
        role: msg.role === 'sent' ? 'user' : 'assistant',
        content: msg.content
      }
    }
  }).reduce((acc, cur) => {
    if (cur) {
      acc.push(cur)
    }
    return acc
  }, [])
  return history.slice(-meta.history_round)
}

const useDatabase = (index) => {
  const selected = opts.databases[index]
  console.log(selected)
  if (index != null && configStore.config.embed_model != selected.embed_model) {
    console.log(selected.embed_model, configStore.config.embed_model)
    message.error(`所选知识库的向量模型（${selected.embed_model}）与当前向量模型（${configStore.config.embed_model}) 不匹配，请重新选择`)
  } else {
    meta.selectedKB = index
  }
}

const handleKeyDown = (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    sendMessage()
  } else if (e.key === 'Enter' && e.shiftKey) {
    // Insert a newline character at the current cursor position
    const textarea = e.target;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    conv.value.inputText.value =
      conv.value.inputText.value.substring(0, start) +
      '\n' +
      conv.value.inputText.value.substring(end);
    nextTick(() => {
      textarea.setSelectionRange(start + 1, start + 1);
    });
  }
}

const renameTitle = () => {
  if (meta.summary_title) {
    const prompt = '请用一个很短的句子关于下面的对话内容的主题起一个名字，不要带标点符号：'
    const firstUserMessage = conv.value.messages[0].content
    const firstAiMessage = conv.value.messages[1].content
    const context = `${prompt}\n\n问题: ${firstUserMessage}\n\n回复: ${firstAiMessage}，主题是（一句话）：`
    simpleCall(context).then((data) => {
      const response = data.response.split("：")[0].replace(/^["'"']/g, '').replace(/["'"']$/g, '')
      emit('rename-title', response)
    })
  } else {
    emit('rename-title', conv.value.messages[0].content)
  }
}

const handleUserScroll = () => {
  // 计算我们是否接近底部（100像素以内）
  const isNearBottom = chatContainer.value.scrollHeight - chatContainer.value.scrollTop - chatContainer.value.clientHeight < 20;

  // 如果用户不在底部，则仅将其标记为用户滚动
  userIsScrolling.value = !isNearBottom;

  // 如果用户再次滚动到底部，请恢复自动滚动
  shouldAutoScroll.value = isNearBottom;
};

const scrollToBottom = () => {
  if (shouldAutoScroll.value) {
    setTimeout(() => {
      chatContainer.value.scrollTop = chatContainer.value.scrollHeight - chatContainer.value.clientHeight;
    }, 10);
  }
}

const generateRandomHash = (length) => {
    let chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    let hash = '';
    for (let i = 0; i < length; i++) {
        hash += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return hash;
}

const appendUserMessage = (msg) => {
  const data = {
    id: generateRandomHash(16),
    role: 'user',
    content: msg
  }
  console.log("👤 添加用户消息：", data)
  conv.value.messages.push(data)
  scrollToBottom()
}


const appendAiMessage = (content, refs=null) => {
  conv.value.messages.push({
    id: generateRandomHash(16),
    role: 'assistant',
    content: content,
    reasoning_content: '',
    refs,
    status: "init",
    meta: {},
    showThinking: "show"
  })
  scrollToBottom()
}

const updateMessage = (info) => {
  const msg = conv.value.messages.find((msg) => msg.id === info.id);
  if (msg) {
    try {
      // 只有在 text 不为空时更新
      if (info.content !== null && info.content !== undefined && info.content !== '') {
        msg.content += info.content;
      }

      if (info.reasoning_content !== null && info.reasoning_content !== undefined && info.reasoning_content !== '') {
        msg.reasoning_content = info.reasoning_content;
      }

      // 只有在 refs 不为空时更新
      if (info.refs !== null && info.refs !== undefined) {
        msg.refs = info.refs;
      }

      if (info.model_name !== null && info.model_name !== undefined && info.model_name !== '') {
        msg.model_name = info.model_name;
      }

      // 只有在 status 不为空时更新
      if (info.status !== null && info.status !== undefined && info.status !== '') {
        msg.status = info.status;
      }

      if (info.meta !== null && info.meta !== undefined) {
        msg.meta = info.meta;
      }

      if (info.message !== null && info.message !== undefined) {
        msg.message = info.message;
      }

      if (info.showThinking !== null && info.showThinking !== undefined) {
        msg.showThinking = info.showThinking;
      }

      // Handle structured data (tool events)
      if (info.data) {
          const evt = info.data;
          if (evt.status === 'tool_start') {
              const toolText = `\n> 🔧 **调用工具**: \`${evt.tool}\`\n> 参数: \`${JSON.stringify(evt.input || {})}\`\n\n`;
              msg.reasoning_content = (msg.reasoning_content || '') + toolText;
          } else if (evt.status === 'tool_end') {
              const toolText = `\n> ✅ **工具完成**: \`${evt.tool}\`\n\n`;
              msg.reasoning_content = (msg.reasoning_content || '') + toolText;
          }
      }

      scrollToBottom();
    } catch (error) {
      console.error('Error updating message:', error);
      msg.status = 'error';
      msg.content = '消息更新失败';
    }
  } else {
    console.error('Message not found:', info.id);
  }
};


const groupRefs = (id) => {
  const msg = conv.value.messages.find((msg) => msg.id === id)
  if (msg.refs && msg.refs.knowledge_base.results.length > 0) {
    msg.groupedResults = msg.refs.knowledge_base.results
        .filter(result => result.file && result.file.filename)
        .reduce((acc, result) => {
          const {filename} = result.file;
          if (!acc[filename]) {
            acc[filename] = []
          }
          acc[filename].push(result)
          return acc;
        }, {})
  }
  scrollToBottom()
}

const simpleCall = (msg) => {
  return new Promise((resolve, reject) => {
    fetch('/api/chat/call', {
      method: 'POST',
      body: JSON.stringify({query: msg,}),
      headers: {'Content-Type': 'application/json'}
    })
        .then((response) => response.json())
        .then((data) => resolve(data))
        .catch((error) => reject(error))
  })
}

const loadDatabases = () => {
  fetch('/api/data/', {method: "GET",})
      .then(response => response.json())
      .then(data => {
        console.log(data)
        opts.databases = data.databases
      })
}

const loadAgents = () => {
  fetch('/api/agents/', {method: "GET",})
      .then(response => response.json())
      .then(data => {
        console.log("Loaded agents:", data)
        opts.agents = data.agents || []
      })
      .catch(err => {
        console.warn("Failed to load agents:", err)
        // 提供默认的 supervisor_agent
        opts.agents = [{ name: "supervisor_agent", description: "智能多工具协调Agent" }]
      })
}

// 新函数用于处理 fetch 请求
const fetchChatResponse = (user_input, cur_res_id) => {
  const controller = new AbortController();
  const signal = controller.signal;

  const params = {
    query: user_input,
    history: getHistory().slice(0, -1), // 去掉最后一条刚添加的用户消息,
    meta: meta,
    cur_res_id: cur_res_id,
  }
  console.log(params)

  console.log(params)

  let url = '/api/chat/';
  // If model_provider is 'agent', route to agent endpoint
  if (configStore.config.model_provider === 'agent') {
      const agentName = configStore.config.model_name;
      url = `/api/chat/agent/${agentName}`;
  }

  fetch(url, {
    method: 'POST',
    body: JSON.stringify(params),
    headers: {
      'Content-Type': 'application/json'
    },
    signal // 添加 signal 用于中断请求
  })
      .then((response) => {
        if (!response.body) throw new Error("ReadableStream not supported.");
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = '';

        const readChunk = () => {
          return reader.read().then(({done, value}) => {
            if (done) {
              const msg = conv.value.messages.find((msg) => msg.id === cur_res_id)
              console.log(msg)
              groupRefs(cur_res_id);
              updateMessage({showThinking: "no", id: cur_res_id});
              isStreaming.value = false;
              if (conv.value.messages.length === 2) {
                renameTitle();
              }
              return;
            }

            buffer += decoder.decode(value, {stream: true});
            const lines = buffer.split('\n');

            // 处理除最后一行外的所有完整行
            for (let i = 0; i < lines.length - 1; i++) {
              const line = lines[i].trim();
              if (line) {
                try {
                  const data = JSON.parse(line);
                  updateMessage({
                    id: cur_res_id,
                    content: data.response,
                    reasoning_content: data.reasoning_content,
                    status: data.status,
                    meta: data.meta,
                    ...data,
                  });
                  // console.log("Last message", conv.value.messages[conv.value.messages.length - 1].content)
                  // console.log("Last message", conv.value.messages[conv.value.messages.length - 1].status)
                  if (data.history && conv.value.messages.length === 0) {
                    conv.value.messages = data.history.map((msg) => ({
                      id: generateRandomHash(8),
                      role: msg.role,
                      content: msg.content
                    }))
                  }

                } catch (e) {
                  console.error('JSON 解析错误:', e, line);
                }
              }
            }

            // 保留最后一个可能不完整的行
            buffer = lines[lines.length - 1];

            return readChunk(); // 继续读取
          });
        };
        readChunk();
      })
      .catch((error) => {
        if (error.name === 'AbortError') {
          console.log('Fetch aborted');
        } else {
          console.error(error);
          updateMessage({
            id: cur_res_id,
            status: "error",
          });
        }
        isStreaming.value = false;
      });

  // 监听 isStreaming 变化，当为 false 时中断请求
  watch(isStreaming, (newValue) => {
    if (!newValue) {
      controller.abort();
    }
  });
}


// 更新后的 sendMessage 函数
const sendMessage = () => {
  const user_input = conv.value.inputText.trim();
  const dbID = opts.databases.length > 0 ? opts.databases[meta.selectedKB]?.db_id : null;
  if (isStreaming.value) {
    message.error('请等待上一条消息处理完成');
    return
  }
  if (user_input) {
    isStreaming.value = true;
    appendUserMessage(user_input);
    appendAiMessage("", null);
    forceScrollToBottom();

    const cur_res_id = conv.value.messages[conv.value.messages.length - 1].id;
    conv.value.inputText = '';
    meta.db_id = dbID;
    meta.mcp_id = meta.use_mcp ? 'default' : null
    fetchChatResponse(user_input, cur_res_id)
  } else {
    console.log('请输入消息');
  }
}

const retryMessage = (id) => {
  // 找到 id 对应的 message，然后删除包含 message 在内以及后面所有的 message
  const index = conv.value.messages.findIndex(msg => msg.id === id);
  const pastMessage = conv.value.messages[index - 1]
  console.log("retryMessage", id, pastMessage)
  conv.value.inputText = pastMessage.content
  if (index !== -1) {
    conv.value.messages = conv.value.messages.slice(0, index - 1);
  }
  console.log(conv.value.messages)
  sendMessage();
}

// 从本地存储加载数据
onMounted(() => {
  scrollToBottom()
  scrollToBottom()
  loadDatabases()
  loadAgents()

  chatContainer.value.addEventListener('scroll', handleUserScroll);

  // 检查现有消息中是否有内容为空的情况
  if (conv.value.messages && conv.value.messages.length > 0) {
    conv.value.messages.forEach(msg => {
      if (msg.role === 'received' && (!msg.content || msg.content.trim() === '')) {
        msg.status = 'error';
        msg.message = '内容加载失败';
      }
    });
  }

  console.log(conv.value.messages)

  // 从本地存储加载数据
  const storedMeta = localStorage.getItem('meta');
  if (storedMeta) {
    const parsedMeta = JSON.parse(storedMeta);
    Object.assign(meta, parsedMeta);
  }
});

onUnmounted(() => {
  if (chatContainer.value) {
    chatContainer.value.removeEventListener('scroll', handleUserScroll);
  }
});

// 添加新函数来处理特定的滚动行为
const forceScrollToBottom = () => {
  shouldAutoScroll.value = true;
  setTimeout(() => {
    chatContainer.value.scrollTop = chatContainer.value.scrollHeight - chatContainer.value.clientHeight;
  }, 10);
};

// 监听 meta 对象的变化，并保存到本地存储
watch(
    () => meta,
    (newMeta) => {
      localStorage.setItem('meta', JSON.stringify(newMeta));
    },
    {deep: true}
);
watch(
    () => meta.themeMode,
    (isDark) => {
      if (isDark) {
        document.body.classList.add('dark-theme');
      } else {
        document.body.classList.remove('dark-theme');
      }
    }
);
// 处理发送或停止
const handleSendOrStop = () => {
  if (isStreaming.value) {
    // 停止生成
    isStreaming.value = false;
    const lastMessage = conv.value.messages[conv.value.messages.length - 1];
    if (lastMessage) {
      lastMessage.isStoppedByUser = true;
      lastMessage.status = 'stopped';
    }
  } else {
    // 发送消息
    sendMessage();
  }
}

// 重试被停止的消息
const retryStoppedMessage = (id) => {
  // 找到用户的原始问题
  const messageIndex = conv.value.messages.findIndex(msg => msg.id === id);
  if (messageIndex > 0) {
    const userMessage = conv.value.messages[messageIndex - 1];
    if (userMessage && userMessage.role === 'sent') {
      conv.value.inputText = userMessage.content;
      // 删除被停止的消息，以及所有后面的消息
      conv.value.messages = conv.value.messages.slice(0, messageIndex - 1);
      // sendMessage();
    }
  }
}

const modelNames = computed(() => configStore.config?.model_names)
const modelStatus = computed(() => configStore.config?.model_provider_status)
const customModels = computed(() => configStore.config?.custom_models || [])

// 筛选 modelStatus 中为真的key
const modelKeys = computed(() => {
  return Object.keys(modelStatus.value || {}).filter(key => modelStatus.value?.[key])
})

// 选择模型的方法
const selectModel = (provider, name) => {
  configStore.setConfigValue('model_provider', provider)
  configStore.setConfigValue('model_name', name)
  message.success(`已切换到模型: ${provider}/${name}`)
}
</script>

<style scoped lang="less">
/* Main Layout */
.chat {
  position: relative;
  width: 100%;
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: var(--background-color);
  overflow: hidden; /* Prevent double scrollbars */

  .chat-header {
    flex: 0 0 64px;
    z-index: 10;
    // ... existing header styles ...
    background-color: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(16px);
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 24px;
    border-bottom: 1px solid var(--border-color);
    
    .header__left, .header__right {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    
    .header__left .close {
        padding: 8px;
        border-radius: 8px;
        cursor: pointer;
        &:hover { background-color: var(--gray-100); }
    }
  }

  /* Empty State / Welcome Screen */
  .welcome-container {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 0 20px;
      gap: 40px;
      
      .logo-container {
          margin-bottom: 24px;
          .welcome-logo {
              width: 80px;
              height: 80px;
              opacity: 0.9;
          }
      }

      .greeting {
          font-size: 32px;
          font-weight: 700;
          color: var(--text-color);
          margin-bottom: 0;
          letter-spacing: -0.02em;
      }
      
      .center-input-wrapper {
          width: 100%;
          max-width: 720px;
          display: flex;
          justify-content: center;
      }
      
      /* Force the input component to have shadow in center mode */
      :deep(.center-input) {
          box-shadow: 0 10px 30px rgba(0,0,0,0.1);
      }

      .suggestion-chips {
          display: flex;
          flex-wrap: wrap;
          justify-content: center;
          gap: 12px;
          max-width: 800px;
          
          .chip {
              padding: 10px 18px;
              background-color: #fff;
              border: 1px solid var(--border-color);
              border-radius: 20px;
              font-size: 13px;
              color: var(--subtext-color);
              cursor: pointer;
              transition: all 0.2s ease;
              display: flex;
              align-items: center;
              gap: 8px;
              box-shadow: 0 2px 4px rgba(0,0,0,0.02);
              
              &:hover {
                  background-color: var(--gray-50);
                  border-color: var(--primary-light-color);
                  color: var(--text-color);
                  transform: translateY(-2px);
              }
          }
      }
  }

  /* Active Chat Layout */
  .chat-main-content {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden; /* Contain inner scroll */
      position: relative;
  }
  
  .chat-box {
    flex: 1;
    overflow-y: auto;
    padding: 24px 0;
    scroll-behavior: smooth;
    display: flex;
    flex-direction: column;
    align-items: center;
    
    &.font-smaller { font-size: 14px; }
    &.font-larger { font-size: 18px; }

    /* Width control for messages */
    :deep(.message-row) {
        width: 100%;
        max-width: 800px; /* ChatGPT width */
        margin: 0 auto;
    }
  }

  /* Bottom Input Area */
  .bottom {
    flex: 0 0 auto;
    padding: 24px;
    background-color: transparent;
    display: flex;
    justify-content: center;
    position: relative;
    
    .message-input-wrapper {
        width: 100%;
        max-width: 800px;
    }
    
    .note {
        text-align: center;
        font-size: 11px;
        color: #999;
        margin-top: 8px;
    }
  }
}

/* Utils */
.nav-btn {
    height: 36px;
    padding: 0 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
    cursor: pointer;
    color: var(--subtext-color);
    transition: all 0.2s;
    
    &:hover {
        background-color: var(--gray-100);
        color: var(--text-color);
    }
    
    &.text {
        font-size: 14px;
        gap: 6px;
    }
}

.action-button {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background-color: var(--surface-card);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
    color: var(--text-color);
    transition: margin 0.2s;
    
    &:hover {
        background-color: var(--gray-50);
    }
}

.model-select {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    cursor: pointer;
    font-weight: 500;
    color: var(--text-color);
    border-radius: 8px;
    &:hover { background-color: var(--gray-100); }
}

/* Animations */
.swing-in-top-fwd {
	animation: swing-in-top-fwd 0.5s cubic-bezier(0.175, 0.885, 0.320, 1.275) both;
}
@keyframes swing-in-top-fwd {
  0% { transform: rotateX(-100deg); transform-origin: top; opacity: 0; }
  100% { transform: rotateX(0deg); transform-origin: top; opacity: 1; }
}

/* Dropdown/Panel Styles */
.my-panal {
    position: absolute;
    width: 320px;
    background: #fff;
    border-radius: 12px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    padding: 16px;
    z-index: 100;
    border: 1px solid var(--border-color);
    top: 60px;
    right: 24px;
    
    .flex-center {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        font-size: 14px;
        color: var(--text-color);
        
        &:last-child { margin-bottom: 0; }
    }
}



/* Settings Panel */
.my-panal {
  position: absolute;
  margin-top: 8px;
  background-color: var(--background-color);
  border: 1px solid var(--border-color);
  box-shadow: var(--shadow-lg);
  border-radius: 12px;
  padding: 16px;
  z-index: 100;
  width: 280px;
  transition: all 0.2s ease;
  color: var(--text-color);

  .flex-center {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    padding: 10px 12px;
    border-radius: 8px;
    transition: background-color 0.2s;
    cursor: pointer;
    font-size: 14px;
    
    &:hover {
        background-color: var(--gray-100);
    }

    .ant-switch {
      &.ant-switch-checked {
        background-color: var(--primary-color);
      }
    }
  }
}

.my-panal.r0.top100 {
  top: 100%;
  right: 0;
}

/* Welcome / Examples Area */
.chat-examples {
  padding: 0 24px;
  text-align: center;
  position: absolute;
  top: 15%; /* Higher up */
  width: 100%;
  z-index: 1;
  animation: slideInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);

  h1 {
    margin-bottom: 32px;
    font-size: 24px;
    font-weight: 600;
    color: var(--text-color);
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
  }
}

.example-cards {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  justify-content: center;
  margin-top: 0;
}

.card {
  position: relative;
  width: 180px;
  height: 120px; /* Shorter cards */
  border-radius: 16px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: var(--shadow-md);
  cursor: pointer;
  transition: all 0.3s ease;
  background-color: white;
  border: 1px solid var(--border-color);
  
  &:hover {
      transform: translateY(-4px);
      box-shadow: var(--shadow-lg);
      border-color: var(--primary-light-color);
      
      .bg {
          color: var(--primary-color);
      }
  }
}

/* Redesigned Card Internals (hiding the old blob/bg structure mostly, making it cleaner) */
.bg {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: var(--input-background-color);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: 500;
  z-index: 2;
  text-align: center;
  padding: 16px;
  color: var(--text-color);
  transition: color 0.2s ease;
}

.blob {
  display: none; /* Hide the childish blob */
}

/* Chat Box Area */
.chat-box {
  width: 100%;
  max-width: 800px;
  margin: 0 auto;
  flex-grow: 1;
  padding: 24px;
  display: flex;
  flex-direction: column;
  transition: max-width 0.3s ease;
  z-index: 2; /* Above background */

  &.wide-screen {
    max-width: 1200px;
  }

  &.font-smaller {
    font-size: 13px;
    .message-box { font-size: 13px; }
  }

  &.font-larger {
    font-size: 16px;
    .message-box { font-size: 16px; }
  }
}

/* Bottom Input Area */
.bottom {
  position: sticky;
  bottom: 0;
  width: 100%;
  margin: 0 auto;
  padding: 12px 24px 24px 24px;
  background: linear-gradient(to top, var(--background-color) 80%, rgba(255,255,255,0));
  z-index: 20;

  .message-input-wrapper {
    width: 100%;
    max-width: 800px;
    margin: 0 auto;
    background-color: var(--input-background-color);
    box-shadow: var(--shadow-lg); /* Floating effect */
    border-radius: 16px;
    border: 1px solid var(--border-color);
    transition: all 0.3s ease;
    padding: 4px; /* Slight padding for inner content */

    &.wide-screen {
      max-width: 1200px;
    }
    
    &:focus-within {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px var(--primary-bg-light), var(--shadow-lg);
    }

    .note {
      width: 100%;
      font-size: 11px;
      text-align: center;
      padding: 0;
      color: var(--subtext-color);
      margin-top: 8px;
      margin-bottom: 0;
      user-select: none;
      opacity: 0.7;
    }
  }
}

.ant-dropdown-link {
  color: var(--text-color);
  cursor: pointer;
}

/* Action Button (New Chat) */
.action-button {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  font-size: 14px;
  font-weight: 500;
  color: #FFFFFF;
  background-color: var(--primary-color);
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: var(--shadow-sm);

  &:hover {
    background-color: var(--primary-hover-color);
    box-shadow: var(--shadow-md);
    transform: translateY(-1px);
  }
  
  &:active {
      transform: translateY(0);
  }

  .icon {
    font-size: 16px;
  }

  .text {
    font-size: 14px;
    white-space: nowrap;
  }
}

/* Model Select */
.model-select {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background-color: var(--input-background-color);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s ease;
  color: var(--text-color);

  &:hover {
    border-color: var(--primary-color);
    color: var(--primary-color);
  }

  .icon {
    font-size: 16px;
    color: var(--primary-color);
  }

  .text {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 160px;
    font-weight: 500;
  }
}

/* Scrollbars */
.chat::-webkit-scrollbar {
  position: absolute;
  width: 6px;
}

.chat::-webkit-scrollbar-track {
  background: transparent;
}

.chat::-webkit-scrollbar-thumb {
  background: var(--gray-300);
  border-radius: 3px;
}

.chat::-webkit-scrollbar-thumb:hover {
  background: var(--gray-400);
}

/* Loading Dots */
.loading-dots {
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.loading-dots div {
  width: 6px;
  height: 6px;
  margin: 0 3px;
  background-color: var(--primary-color); /* Updated color */
  border-radius: 50%;
  opacity: 0.5;
  animation: pulse 0.6s infinite ease-in-out both;
}

.loading-dots div:nth-child(1) { animation-delay: -0.32s; }
.loading-dots div:nth-child(2) { animation-delay: -0.16s; }

@keyframes pulse {
  0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
  40% { transform: scale(1.2); opacity: 1; }
}

@keyframes slideInUp {
  from { transform: translateY(40px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

/* Mobile */
@media (max-width: 520px) {
  .chat {
    height: calc(100vh - 64px); /* Match header height */
  }

  .chat-container .chat .chat-header {
    background: var(--background-color);

    .header__left, .header__right {
      gap: 12px;
    }

    .nav-btn {
      padding: 0;
      .text { display: none; }
    }
  }

  .bottom {
    padding: 8px 12px;
    
    .message-input-wrapper {
        border-radius: 12px;
    }
    
    .note { display: none; }
  }
}

.scrollable-menu {
  max-height: 300px;
  overflow-y: auto;

  &::-webkit-scrollbar { width: 6px; }
  &::-webkit-scrollbar-track { background: transparent; }
  &::-webkit-scrollbar-thumb { background: var(--gray-300); border-radius: 3px; }
}
</style>

<style lang="less">
// Global styles for dropdown
.ant-dropdown-menu {
  border-radius: 12px !important;
  box-shadow: var(--shadow-lg) !important;
  padding: 8px !important;
  
  &.scrollable-menu {
    max-height: 300px;
    overflow-y: auto;
  }
  
  .ant-dropdown-menu-item {
      border-radius: 8px !important;
      padding: 8px 12px !important;
      
      &:hover {
          background-color: var(--gray-100) !important;
          color: var(--primary-color) !important;
      }
  }
  
  .ant-dropdown-menu-item-group-title {
      font-size: 12px;
      color: var(--subtext-color);
  }
}
</style>

