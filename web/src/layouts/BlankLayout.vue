<template>
  <div>
<!--    我从 <router-view> 里拿到了当前路由匹配的组件，并把它叫做 Component。-->
    <router-view v-slot="{ Component }">
<!--      缓存组件实例，防止它们被销毁，再次显示时可以直接复用之前的状态。-->
      <keep-alive>
        <component :is="Component" />
      </keep-alive>
    </router-view>
  </div>
</template>
