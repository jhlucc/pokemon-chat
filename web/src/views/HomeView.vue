<template>
  <section class="welcome" ref="root">
    <header class="app-header">
      <h1 class="visually-hidden">{{ resolvedTitle }}</h1>
    </header>

    <main class="hero">
      <h2 class="hero__title" v-text="resolvedTitle" />
      <p class="hero__subtitle">大模型驱动的知识库管理工具</p>
      <button class="cta" @click="goToChat" aria-label="开始对话">
        开始对话
      </button>
      <img class="hero__img" src="/home.jpg" alt="彩色宝可梦知识图谱插图" loading="lazy" />
    </main>

    <footer class="app-footer">
      © {{ new Date().getFullYear() }} Poké Knowledge. All rights reserved.
    </footer>
  </section>
</template>

<script setup>
import { computed } from 'vue'
import { useRouter } from 'vue-router'

/**
 * Optional title prop so页面可复用
 */
const props = defineProps({
  title: { type: String, default: '可萌助手' }
})
const resolvedTitle = computed(() => props.title)

const router = useRouter()
const goToChat = () => router.push('/chat')
</script>

<style scoped lang="less">
/*  🎨 Variables  ---------------------------------------------------------- */
@primary       : #2c86a8;
@primary-dark  : #005f77;
@text-color    : #333;
@gradient      : linear-gradient(168deg,#ffd6eb,#ffe7ca,#d3fffb,#dbebff,#ffd8ff);

/*  🌐 Layout  ------------------------------------------------------------- */
.welcome {
  --header-height: 64px;
  display:flex;
  flex-direction:column;
  align-items:center;
  min-height:100vh;
  color:@text-color;
  text-align:center;
  background:@gradient;
  background-size:1000% 1000%;
  animation:bgMove 18s ease-in-out infinite;
}

/* reduce‑motion 优化 */
@media (prefers-reduced-motion: reduce) {
  .welcome { animation:none; }
}

.app-header {
  height:var(--header-height);
  width:100%;
  display:flex;
  align-items:center;
  justify-content:center;
  backdrop-filter:blur(12px);
  background:rgba(255,255,255,.45);
  border-bottom:1px solid fade(@primary,30%);
}

.hero {
  flex:1 0 auto;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  width:100%;
  padding:clamp(1rem,4vw,4rem);
}

.hero__title {
  font-size:clamp(2.5rem,6vw,3.75rem);
  font-weight:700;
  margin:0 0 .5rem;
}

.hero__subtitle {
  font-size:clamp(1rem,2.2vw,1.25rem);
  margin:0 0 1.75rem;
}

.cta {
  padding:.65rem 2.5rem;
  font-size:clamp(1rem,2.4vw,1.35rem);
  font-weight:600;
  color:#fff;
  background:@primary-dark;
  border:none;
  border-radius:999px;
  cursor:pointer;
  transition:all .25s ease;

  &:hover {
    background:darken(@primary-dark,6%);
    transform:translateY(-2px);
    box-shadow:0 4px 12px rgba(0,0,0,.15);
  }
}

.hero__img {
  width:min(720px,90%);
  height:auto;
  margin-top:clamp(1.5rem,5vw,3rem);
  border-radius:1rem;
  object-fit:cover;
  box-shadow:0 5px 12px rgba(0,0,0,.08);
}

.app-footer {
  padding:.75rem 0 1.25rem;
  font-size:.875rem;
  color:fade(@text-color,60%);
}

/*  💫 Background animation  ---------------------------------------------- */
@keyframes bgMove {
  0%{background-position:0% 50%}
  50%{background-position:100% 50%}
  100%{background-position:0% 50%}
}

/*  ♿ 隐藏但可读标题  ------------------------------------------------------ */
.visually-hidden {
  border:0;clip:rect(0 0 0 0);height:1px;margin:-1px;overflow:hidden;padding:0;position:absolute;width:1px;
}
</style>
