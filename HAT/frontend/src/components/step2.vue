<template>
	<div class="step2">
		<!-- UI controls that can are used to manipulate the display of the chart -->

		<div v-if="loading">
			<p>Current progress:
				<span>{{loadingprogress}}</span>
			</p>
			<p>Current uptime:
				<span>{{uptime}}</span>
			</p>
		</div>

		<div style="height:100vh;width:100%;display:flex;flex-direction:row">
			<div style="flex:1;overflow:auto">

				<tree
				 :data="tree"
				 node-text="value"
				 layout-type="circular"
				 type="cluster"
				 :duration="vued3treeduration"
				 :zoomable="vued3treezoomable"
				 :identifier="identifierGetter"
				 style="max-height:95vh;height:95vh;"
				 @clicked="lookAtNodeInfo"
				 @mouseNodeOver="testFunc"
				/>
			</div>

			<div style="flex-basis:18%;display:flex;flex-direction:column;">
				<div
				 style="flex:28 0;border:solid"
				 v-html="currentNodeInfo"
				>

				</div>
				<div style="flex:28 0;border:solid">
					<ol>
						<li
						 v-for="(cluster_en,idx) in similarCluster"
						 :key="idx"
						>
							<span>{{cluster_en[0]}}</span>: <span>{{cluster_en[1]}}</span>
						</li>
					</ol>
				</div>
			</div>

		</div>

	</div>
</template>

<script>
import { hierarchicalEdgeBundling, tree } from 'vued3tree';
import axios from 'axios';
import constants from '@/constants.js'
export default {
	name: 'step2',
	data() {
		return {
			csv: null,
			internalselected: [],
			search: "",
			settings: {
				strokeColor: "#19B5FF",
				width: 2400,
				height: 1200
			},
			clusters: [],
			loading: false,
			loadingprogress: "Connecting to server",
			uptime: 0,
			checkingtimeout: null,
			existingClusters: [],
			jumptoitselfofstep3: false,
			currentNodeInfo: "",
			similarCluster: [],
			tree: {
				name: "father",
				children: [{
					name: "son1",
					children: [{ name: "grandson", id: 1 }, { name: "grandson2", id: 2 }]
				}, {
					name: "son2",
					children: [{ name: "grandson3", id: 3 }, { name: "grandson4", id: 4 }]
				}]
			},
			links: [
				{ source: 3, target: 1, type: 1 },
				{ source: 3, target: 4, type: 2 }
			],
			linkTypes: [
				{ id: 1, name: 'depends', symmetric: true },
				{ id: 2, name: 'implement', inName: 'implements', outName: 'is implemented by' },
				{ id: 3, name: 'uses', inName: 'uses', outName: 'is used by' },
			],
			vued3treeduration: 1,
			vued3treezoomable: true
		}
	},
	components: {
		hierarchicalEdgeBundling,
		tree
	},
	computed: {


	},
	mounted: function () {
		// this.checkprogress();
		this.loadHACTree();
	},
	methods: {
		dfsGetAllChildNode: function (node, currentNodeSet) {
			const self = this;
			if (node.bert_emb_idx) {
				currentNodeSet.add(node.bert_emb_idx);
			}
			if (node.children) {
				for (let i = 0; i < node.children.length; i++) {
					self.dfsGetAllChildNode(node.children[i], currentNodeSet);
				}
			}

		},
		loadHACTree: function () {
			const self = this;
			axios({ 'baseURL': constants.baseURL, 'url': '/s2/3', 'method': 'GET', "params": { "session": self.$localStorage.get('session', 'dummy') } }).then(resp => {
				self.csv = resp.data.csv;
				self.tree = resp.data.tree;
				self.links = resp.data.links;
				self.existingClusters = resp.data.clusters;
			}, err => {
				console.log(err);
			});
		},
		lookAtNodeInfo(node) {
			const self = this;
			if (node.data.bert_emb_idx !== null) {
				self.currentNodeInfo = node.data.sentence;
			}
			else {
				self.currentNodeInfo = ""
			}
			const getSimilarCluster = function () {
				const subTreeNodes = new Set();
				self.dfsGetAllChildNode(node.data, subTreeNodes);
				axios({					'baseURL': constants.baseURL, 'url': '/s2/get_similar_eventtypes', 'method': 'POST', "params": { "session": self.$localStorage.get('session', 'dummy') }, "data": {
						"event_mention_emb_ids": Array.from(subTreeNodes)
					}				}).then(resp => {
					const ret = resp.data.ret;
					const sortingArr = [];
					for (const [key, value] of Object.entries(ret)) {
						sortingArr.push([key, value]);
					}
					self.similarCluster = sortingArr.sort((a, b) => { return a[1] - b[1]; }).slice(0, Math.min(5, sortingArr.length));
				}, err => {
					console.log(err);
				});
			};
			getSimilarCluster();
		},
		getHistory: function () {
			const that = this;
			this.checkingtimeout && clearTimeout(this.checkingtimeout);
			that.loading = false;
			axios({ 'baseURL': constants.baseURL, 'url': '/s2/3', 'method': 'GET', "params": { "session": that.$localStorage.get('session', 'dummy') } }).then(resp => {
				that.csv = resp.data.csv;
				that.settings.height = resp.data.cnt * 18;
				that.existingClusters = resp.data.clusters;
			}, err => {
				console.log(err);
			});
		},
		select: function (index, node) {
			this.csv = this.csv.filter((item, idx) => {
				if (item.id.includes(node.id)) {
					this.internalselected.push({ idx: idx, data: item });
					if (item.bert_emb_idx) {
						axios({ 'baseURL': constants.baseURL, 'url': '/s2/blacklist_nodes', 'method': 'POST', "params": { "session": this.$localStorage.get('session', 'dummy') }, "data": { "csvs": [item] } }).then(resp => { }, err => { console.log(err) });
					}
					return false;
				}
				else {
					return true;
				}
			})
		},
		identifierGetter: function (node) {
			return node.id;
		},
		testFunc:function(node){
			console.log(node);
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
