<template>
    <div class="index">
        <input v-model="session" @change="changeSession($event)"/>
        <button @click="gotoStep(3)">Go To Event Type Customization Step</button>
        <!-- <button @click="gotoStep(4)">Go To Adjudication Step</button> -->
        <!-- <button @click="gotoStep4PendingPage()">GoTo Step4</button> -->
    </div>
</template>

<script>
export default {
    name: 'index',
    data() {
        return {
            session:""
        }
    },
    props: [],
    mounted() {
        this.session = this.$localStorage.get('session','dummy');
    },
    methods: {
        changeSession:function(evt){
            this.session = evt.target.value;
            this.$localStorage.set('session',this.session);
        },
        gotoStep:function(step){
            this.$router.replace({ name: 'step'+step });
        },
        gotoStep4PendingPage:function(){
            this.$router.replace({name:"pending_page_for_step_4"});
        }
    },

}
</script>

<style scoped>
.index {
}
</style>
