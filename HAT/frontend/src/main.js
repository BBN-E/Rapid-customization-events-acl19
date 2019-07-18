// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import SuiVue from 'semantic-ui-vue'
Vue.use(SuiVue);
import VueLocalStorage from 'vue-localstorage'
Vue.use(VueLocalStorage);
import App from './App'
import router from './router'

Vue.config.productionTip = true
import 'semantic-ui-css/semantic.min.css'

import Element from 'element-ui'

Vue.use(Element)
import 'element-ui/lib/theme-chalk/index.css';
/* eslint-disable no-new */
const vm = new Vue({
  el: '#app',
  router,
  components: { App },
  template: '<App/>'
})
