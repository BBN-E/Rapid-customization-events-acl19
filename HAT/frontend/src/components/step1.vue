<template>
	<div class="step1">
		<div id="control">
			<button :disabled='onLoading' @click="init()">Init</button>
			<input type="number" v-model="wordFrequencyThreshold" placeholder="word frequency threshold" />
			<input type="number" v-model="phraseFrequencyThreshold" placeholder="phrase Frequency threshold" />
			<button :disabled='onLoading' @click="getCorpus()">Process Corpus</button>
			<button :disabled='onLoading' @click="gotoStep2()">Finish Cleaning</button>
		</div>
		<div id="mainframe">
			<!-- <sui-list>
				<sui-list-item v-for="(item,idx) in corpus" :key="idx">
					{{item.trigger_lemma}} {{item.trigger_postag}} {{item.cnt}}
				</sui-list-item>
			</sui-list> -->
			<vuetable ref="vuetable" :api-mode="false" :data="internalcorpus" :fields="fields" data-path="data" pagination-path="links.pagination" @vuetable:pagination-data="onPaginationData" :data-manager="dataManager">
				<template slot="actions" slot-scope="props">
					<button @click="toggleWord(props.rowData,props.rowIndex)">Toggle</button>
					<span style="color:red" v-if="props.rowData.inStopwordList">Deleted</span>
				</template>
			</vuetable>
		</div>
		<div id="pagination">
			<vuetable-pagination ref="pagination" @vuetable-pagination:change-page="onChangePage"></vuetable-pagination>
		</div>
	</div>
</template>

<script>
import axios from 'axios'
import Vuetable from 'vuetable-2/src/components/Vuetable'
import VuetablePagination from 'vuetable-2/src/components/VuetablePagination'
import constants from '@/constants.js'
export default {
	name: 'step1',
	data() {
		return {
			internalcorpus: {
				"data": [], "links": {
					"pagination": {
						"total": 0,
						"per_page": 15,
						"current_page": 1,
						"last_page": 1,
						"next_page_url": null,
						"prev_page_url": null,
						"from": 0,
						"to": 0,
					}
				},			},
			onLoading: false,
			fields: ["trigger", "trigger_postag", "cnt", "__slot:actions"],
			stopword: new Set(),
			wordFrequencyThreshold:-1,
			phraseFrequencyThreshold:-1,
		};
	},
	methods: {
		dataManager(sortOrder, pagination) {
			const self = this;
			return axios({'baseURL':constants.baseURL, "url": "/s1page/" + (pagination.current_page-1) + ".json", "method": "GET","params":{"session":self.$localStorage.get('session','dummy')} }).then(resp => {
				resp.data.data = resp.data.data.map((item) => {
					item['inStopwordList'] = this.inStopwordList(item);
					return item;
				});
				resp.data.links.pagination.next_page_url = constants.baseURL + resp.data.links.pagination.next_page_url;
				resp.data.links.pagination.prev_page_url = constants.baseURL + resp.data.links.pagination.prev_page_url;
				this.internalcorpus = resp.data;
				return this.internalcorpus;
			}, err => {
				console.log(err);
				return this.internalcorpus;
			});
		},
		onChangePage(page) {
			this.$refs.vuetable.changePage(page);
		},
		onPaginationData(paginationData) {
			this.$refs.pagination.setPaginationData(paginationData);
		},
		inStopwordList: function (item) {
			return this.stopword.has(item.id);
		},
		formatWord: function (val) {

		},
		toggleWord: function (data, idx) {
			if (this.stopword.has(data.id)) {
				this.stopword.delete(data.id);
				data.inStopwordList = false;
			}
			else {
				this.stopword.add(data.id);
				data.inStopwordList = true;
			}
		},
		getCorpus: function () {
			const self = this;
			this.onLoading = true;
			this.internalcorpus = {
				"data": [], "links": {
					"pagination": {
						"total": 0,
						"per_page": 15,
						"current_page": 1,
						"last_page": 1,
						"next_page_url": null,
						"prev_page_url": null,
						"from": 0,
						"to": 0,
					}
				},			};
			axios({ 'baseURL':constants.baseURL,"url": "/s1", "method": "POST", "data": { "stopword": Array.from(this.stopword),"frequency_word":this.wordFrequencyThreshold,'frequency_phrase': this.phraseFrequencyThreshold} ,"params":{"session":self.$localStorage.get('session','dummy')}}).then(resp => {
				resp.data.data = resp.data.data.map((item) => {
					item['inStopwordList'] = this.inStopwordList(item);
					return item;
				});
				resp.data.links.pagination.next_page_url = constants.baseURL + resp.data.links.pagination.next_page_url;
				resp.data.links.pagination.prev_page_url = constants.baseURL + resp.data.links.pagination.prev_page_url;
				this.internalcorpus = resp.data;
				this.onLoading = false;
				this.stopword = new Set();
			},
				err => {
					console.log(err);
					this.onLoading = false;
					this.stopword = new Set();
				});
			
		},
		gotoStep2: function () {
			const self = this;
			this.onLoading = true;
						this.internalcorpus = {
				"data": [], "links": {
					"pagination": {
						"total": 0,
						"per_page": 15,
						"current_page": 1,
						"last_page": 1,
						"next_page_url": null,
						"prev_page_url": null,
						"from": 0,
						"to": 0,
					}
				},			};
			axios({ 'baseURL':constants.baseURL, 'url':'/s2/fromstep1','method':'POST','data':{ "stopword": Array.from(this.stopword),"frequency_word":this.wordFrequencyThreshold,'frequency_phrase': this.phraseFrequencyThreshold },"params":{"session":self.$localStorage.get('session','dummy')}}).then(resp=>{
				this.$router.replace({ path: '/s2' });
			},err=>{
				console.log(err);
			});
		},
		init: function () {
			const self = this;
			this.onLoading = true;
			this.wordFrequencyThreshold=-1;
			this.phraseFrequencyThreshold=-1;
						this.internalcorpus = {
				"data": [], "links": {
					"pagination": {
						"total": 0,
						"per_page": 15,
						"current_page": 1,
						"last_page": 1,
						"next_page_url": null,
						"prev_page_url": null,
						"from": 0,
						"to": 0,
					}
				},			};
			axios({ 'baseURL':constants.baseURL,"url": "/s1/reset", "method": "POST" ,"params":{"session":self.$localStorage.get('session','dummy')}}).then(resp => {
				resp.data.data = resp.data.data.map((item) => {
					item['inStopwordList'] = this.inStopwordList(item);
					return item;
				});
				resp.data.links.pagination.next_page_url = constants.baseURL + resp.data.links.pagination.next_page_url;
				resp.data.links.pagination.prev_page_url = constants.baseURL + resp.data.links.pagination.prev_page_url;
				this.internalcorpus = resp.data;
				this.onLoading = false;
				this.stopword = new Set();
			},
				err => {
					console.log(err);
					this.onLoading = false;
				});
		}
	},
	mounted() {
		// this.getCorpus();
	},
	components: {
		Vuetable,
		VuetablePagination
	},
	computed: {

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
.step1{
	height: 100%;
	display:flex;
	flex-direction: column;
	justify-content: space-around;
}
#control{
	flex:1 0 5%;
}
#mainframe{
	flex:1 1 90%;
}
#pagination{
	flex: 1 0 5%;
}
</style>
