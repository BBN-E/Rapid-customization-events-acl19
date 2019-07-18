import Vue from 'vue'
import Router from 'vue-router'
import step2 from '@/components/step2'
import step3 from '@/components/step3'
import step4 from '@/components/step4'
import index from '@/components/index'
import pending from '@/components/pending'
import pending_page_for_step_4 from '@/components/pending_page_for_step_4'
Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/s2',
      name: 'step2',
      component: step2
    },
    {
      path: '/s3',
      name: 'step3',
      component: step3
    },
    {
      path: '/s4',
      name: 'step4',
      component: step4
    },
    {
      path: '/',
      name: 'index',
      component: index
    },
    {
      path:'/pending_page_for_step_4',
      name:'pending_page_for_step_4',
      component:pending_page_for_step_4
    }
  ]
})
