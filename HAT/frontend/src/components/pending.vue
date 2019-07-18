<template>
    <div class="pending">
        <!-- UI controls that can are used to manipulate the display of the chart -->

        <div v-if="loading">
            <p>Current progress:
                <span>{{loadingprogress}}</span>
            </p>
            <p>Current uptime:
                <span>{{uptime}}</span>
            </p>
        </div>

    </div>
</template>

<script>
import axios from 'axios';
import constants from '@/constants.js'
export default {
    name: 'pending',
    data() {
        return {
            loading: true,
            loadingprogress: "Connecting to server",
            uptime: 0,
            checkingtimeout: null
        }
    },
    components: {

    },
    computed: {

    },
    mounted: function () {
        this.checkprogress();
    },
    methods: {
        checkprogress: function () {
            const that = this;
            const check = function () {
                axios({ 'baseURL': constants.baseURL, 'url': '/s2/2', 'method': 'GET', "params": { "session": that.$localStorage.get('session', 'dummy') } }).then(resp => {
                    that.loadingprogress = resp.data.progress;
                    that.uptime = resp.data.uptime;
                    if (typeof resp.data.redirect !== "undefined" && resp.data.redirect !== null) {
                        if(typeof resp.data.exception !== "undefined" && resp.data.exception !== null){
                            alert(resp.data.exception);
                        }
                        that.$router.replace({ name: resp.data.redirect });
                    }
                    else {
                        if (that.loading) {
                            that.checkingtimeout = setTimeout(check, 1000);
                        }
                    }
                }, err => {
                    console.log(err);
                })
            }
            this.checkingtimeout = setTimeout(check, 1000);
        }
    }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h1,
h2 {
  font-weight: normal;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
#hello {
  height: 100%;
  width: 100%;
}

.node {
  opacity: 1;
}

.node circle {
  fill: #999;
  cursor: pointer;
}

.node text {
  font: 16px sans-serif;
  cursor: pointer;
}

.node--internal circle {
  fill: #555;
}

.node--internal text {
  text-shadow: 0 1px 0 #fff, 0 -1px 0 #fff, 1px 0 0 #fff, -1px 0 0 #fff;
}

.link {
  fill: none;
  stroke: #555;
  stroke-opacity: 0.4;
  stroke-width: 1.5px;
  stroke-dasharray: 1000;
}

.node:hover {
  pointer-events: all;
  stroke: #ff0000;
}

.node.highlight {
  fill: red;
}

.controls {
  position: fixed;
  top: 16px;
  left: 16px;
  background: #f8f8f8;
  padding: 0.5rem;
  display: flex;
  flex-direction: column;
  max-width: 300px;
  max-height: 300px;
  /* overflow-y: scroll; */
}

.controls > * + * {
  margin-top: 1rem;
}

label {
  display: block;
}

.node.existtrigger {
  fill: green;
}
</style>
