<template>
  <div id="step3">
    <div id="navigate">
      <button
        style="flex:1"
        @click="toggleModalPanel('workingClusterSelectorOn')"
      >Select Display Cluster</button>
      <button style="flex:1" @click="saveAndClearBoard()">Save And Clear</button>
      <button style="flex:1" @click="taintAndSave()">Save</button>
      <!-- <button style="flex:1" @click="saveAndStartTraining()">Save And Training</button> -->
    </div>
    <div id="main">
      <sui-card
        id="current"
        v-if="!similarClusterOn"
        style="margin:0;background-color:#f4f5f7;margin:0;flex:0 0 auto"
      >
        <sui-card-content style="display:flex;flex-direction:column;overflow-y:auto">
          <sui-card-header>
            <i class="icon"></i>
            <div class="ui transparent input">
              <input
                type="text"
                placeholder="Cluster Name"
                @change="focusClusterNameChangeHandler($event)"
              >
            </div>
            <button :disabled="!focusCluster" @click="toggleAddTriggerPanelOn(focusCluster)">Add new</button>
            <button
              :disabled="!focusCluster"
              @click="showAnnotatedExample(focusCluster)"
            >Show known examples</button>
            <button
              :disabled="!focusCluster"
              @click="hideAnnotatedExample(focusCluster)"
            >Hide known examples</button>
            <!-- <button :disabled="!focusCluster" @click="popupGroundingPanel(focusCluster)">Grounding</button> -->
          </sui-card-header>
          <sui-card-description style="flex:1;overflow-y:auto;display:flex;flex-direction:column;">
            <local-draggable
              v-if="focusCluster"
              style="flex:1;overflow-y:auto"
              :root="focusCluster"
              :movecallback="checkMove"
              :eventhandlers="eventhandler"
            ></local-draggable>
          </sui-card-description>
        </sui-card-content>
      </sui-card>
      <sui-card
        id="similarcluster"
        v-if="similarClusterOn"
        style="margin:0;background-color:#e6ffe6;margin:0;flex:0 0 auto"
      >
        <sui-card-content style="display:flex;flex-direction:column;overflow-y:auto">
          <sui-card-header>
            <i class="icon"></i>
            <div class="ui transparent input">
              <input
                type="text"
                placeholder="Similar Cluster"
                :value="similarclusterdisplaystring"
                disabled
                class="disabledBox"
              >
            </div>
            <button @click="showSimilarCluster(null)">Close</button>
          </sui-card-header>
          <sui-card-description style="flex:1;overflow-y:auto;display:flex;flex-direction:column;">
            <local-draggable
              style="flex:1;overflow-y:auto"
              :root="similarCluster"
              :movecallback="checkMove"
              :eventhandlers="eventhandler"
            ></local-draggable>
          </sui-card-description>
        </sui-card-content>
      </sui-card>
      <div id="clusters">
        <sui-card v-for="(el,idx) in workingCluster" :key="idx" style="margin:0;flex:0 0 auto">
          <sui-card-content style="display:flex;flex-direction:column;overflow-y:auto">
            <sui-card-header>
              <i
                class="minus icon"
                :class="{'touched':el.aux && !el.aux.touched}"
                @click="eventhandler.toggleAnnotation(el)"
              ></i>
              <i class="toggle on icon" @click="hideCluster(el)"></i>
              <div class="ui transparent input">
                <input
                  type="text"
                  placeholder="Cluster Name"
                  :value="el.key"
                  @change="clusterNameChangeChecker(el,$event)"
                >
              </div>
              <button @click="toggleAddTriggerPanelOn(el)">Add new</button>
              <button @click="showSimilarCluster(el)">Find similar</button>
              <button @click="showAnnotatedExample(el)">Show known examples</button>
              <button @click="hideAnnotatedExample(el)">Hide known examples</button>
              <!-- <button @click="popupGroundingPanel(el)">Grounding</button> -->
            </sui-card-header>
            <sui-card-description
              style="flex:1;overflow-y:auto;display:flex;flex-direction:column;"
            >
              <local-draggable
                style="flex:1;overflow-y:auto"
                :root="el"
                :movecallback="checkMove"
                :eventhandlers="eventhandler"
              ></local-draggable>
            </sui-card-description>
          </sui-card-content>
        </sui-card>
      </div>
    </div>
    <sui-modal v-model="workingClusterSelectorOn">
      <sui-modal-header>Select Working Clusters</sui-modal-header>
      <sui-modal-content class="scrolling">
        <sui-modal-description>
          <button @click="selectDisplayCluster(true)">Select All</button>
          <button @click="selectDisplayCluster(false)">Clear All</button>
          <sui-item-group>
            <sui-item v-for="(el,idx) in clusters.children" :key="idx">
              <sui-item-content>
                <sui-item-meta style="display:flex">
                  <sui-checkbox v-model="el.aux.working"/>
                  <input
                    type="text"
                    style="flex:1"
                    placeholder="Cluster Name"
                    :value="el.key"
                    @change="clusterNameChangeChecker(el,$event)"
                  >
                  <!-- <i class="minus icon" @click="eventhandler.toggleAnnotation(el)"></i> -->
                </sui-item-meta>
              </sui-item-content>
            </sui-item>
          </sui-item-group>
        </sui-modal-description>
      </sui-modal-content>
      <sui-modal-actions>
        <sui-button
          floated="right"
          positive
          @click.native="toggleModalPanel('workingClusterSelectorOn')"
        >OK</sui-button>
      </sui-modal-actions>
    </sui-modal>
    <sui-modal v-model="addTriggerPanelOn">
      <sui-modal-header>Add Trigger</sui-modal-header>
      <sui-modal-content>
        <sui-modal-description>
          <input
            type="text"
            style="flex:1"
            placeholder="Trigger Lemma"
            v-model="triggerLemmaForAddingTrigger"
            :disabled="triggerPrefilled"
          >
          <input
            type="text"
            style="flex:1"
            placeholder="Full Text Search Key"
            v-model="addTriggerFullTextSearchKey"
          >
        </sui-modal-description>
      </sui-modal-content>
      <sui-modal-actions>
        <sui-button floated="right" positive @click.native="searchTrigger">Search and Add</sui-button>
        <sui-button
          floated="right"
          positive
          @click.native="toggleModalPanel('addTriggerPanelOn')"
        >Close</sui-button>
      </sui-modal-actions>
    </sui-modal>
    <sui-modal v-model="groundingSelectorOn">
      <sui-modal-header>Grounding</sui-modal-header>
      <sui-modal-content>
        <sui-modal-description>
          <sui-item-group>
            <sui-item v-for="(el,idx) in groundingCandidates" :key="idx">
              <sui-item-content>
                <sui-item-meta style="display:flex">
                  <input
                    type="radio"
                    name="selectedGroundingType"
                    :value="el"
                    v-model="userGroundingDecision"
                  >
                  {{el}}
                  <br>
                </sui-item-meta>
              </sui-item-content>
            </sui-item>
            <sui-item>
              <sui-item-content>
                <sui-item-meta style="display:flex">
                  <input type="radio" name="selectedGroundingType" v-model="userGroundingDecision">
                  <input type="text" v-model="userGroundingDecision">
                </sui-item-meta>
              </sui-item-content>
            </sui-item>
          </sui-item-group>
        </sui-modal-description>
      </sui-modal-content>
      <sui-modal-actions>
        <sui-button floated="right" positive @click.native="submitGroundingDecision">Save</sui-button>
        <sui-button
          floated="right"
          positive
          @click.native="toggleModalPanel('groundingSelectorOn')"
        >Close</sui-button>
      </sui-modal-actions>
    </sui-modal>
  </div>
</template>

<script>
import draggable from "vuedraggable";
import local from "./localdraggable.vue";
import axios from "axios";
import constants from "@/constants.js";
import _ from "lodash";
export default {
  name: "step3",
  methods: {
    toggleAddTriggerPanelOn: function(el) {
      const self = this;
      let curCluster;
      if (el.type === "trigger") {
        curCluster = el.parent;
        self.triggerPrefilled = true;
        self.triggerLemmaForAddingTrigger = el.key;
      } else {
        curCluster = el;
        self.triggerPrefilled = false;
        self.triggerLemmaForAddingTrigger = "";
      }
      self.addTriggerFullTextSearchKey = "";
      self.addTriggerFocusCluster = curCluster;
      self.addTriggerPanelOn = true;
    },
    searchTrigger: function() {
      const self = this;
      const trigger_lemma = self.triggerLemmaForAddingTrigger;
      const curCluster = self.addTriggerFocusCluster;
      let potentialTriggerNode = {
        key: trigger_lemma,
        type: "trigger",
        children: [],
        aux: {
          touched: false,
          blacklist: [],
          trigger: trigger_lemma,
          trigger_postag: null,
          fullTextSearchkey: self.addTriggerFullTextSearchKey
        },
        parent: curCluster
      };
      let found = false;
      for (let i = 0; i < curCluster.children.length; ++i) {
        if (curCluster.children[i].key === trigger_lemma) {
          potentialTriggerNode = curCluster.children[i];
          found = true;
          potentialTriggerNode["aux"]["fullTextSearchkey"] =
            self.addTriggerFullTextSearchKey;
          break;
        }
      }
      const tmpBlackList = Array.concat(
        potentialTriggerNode.aux.blacklist,
        potentialTriggerNode.children.map(sentence => {
          return sentence.aux.instanceId;
        })
      );
      const eventType = curCluster.key;
      const triggers = [
        {
          trigger: trigger_lemma,
          postag: null,
          blacklist: tmpBlackList,
          fullTextSearchStr: self.addTriggerFullTextSearchKey
        }
      ];
      axios({
        baseURL: constants.baseURL,
        url: "/s3/query_unannotated_sentence",
        method: "POST",
        data: { eventType: curCluster.key, triggers: triggers },
        params: { session: self.$localStorage.get("session", "dummy") }
      }).then(
        resp => {
          const ret = resp.data;
          const touched_root_root = potentialTriggerNode.parent;
          const dfs_mark_touch = function(root) {
            root.aux.touched = true;
            if (typeof root.children !== "undefined") {
              for (let i = 0; i < root.children.length; ++i) {
                dfs_mark_touch(root.children[i]);
              }
            }
          };
          dfs_mark_touch(touched_root_root);
          ret.map(trigger => {
            trigger.children.map(sentence => {
              sentence.parent = potentialTriggerNode;
              potentialTriggerNode.children.push(sentence);
            });
          });
          if (!found && ret[0].children.length > 0) {
            curCluster.children.push(potentialTriggerNode);
          }
          if (ret[0].children.length > 0) {
            self.addTriggerPanelOn = false;
          } else {
            alert("Cannot find anything.");
          }
        },
        err => {
          console.log(err);
        }
      );
    },
    showSimilarCluster: function(cluster) {
      const self = this;
      self.similarClusterOn = false;
      if (!cluster) return;
      const curTrigger = cluster["children"].map(item => {
        return item.key;
      });
      self.similarCluster = { type: "root", children: [] };
      axios({
        baseURL: constants.baseURL,
        //url: "/v2/s2/similarcluster",
        url: "/s3/similarcluster",
        method: "POST",
        data: { triggers: curTrigger, threshold: -1, cutoff: 10 },
        params: { session: self.$localStorage.get("session", "dummy") }
      }).then(
        resp => {
          console.log(resp);
          resp.data.children = resp.data.children.map(trigger => {
            trigger.parent = resp.data;
            if (trigger.children) {
              trigger.children = trigger.children.map(sentence => {
                sentence.parent = trigger;
                return sentence;
              });
            } else {
              trigger.children = [];
            }
            if (typeof trigger.aux !== "object") trigger.aux = {};
            trigger.aux.trigger = trigger.key;
            trigger.aux.blacklist = [];
            return trigger;
          });
          resp.data.parent = null;
          self.similarCluster = resp.data;
          self.similarClusterOn = true;
        },
        err => {
          console.log(err);
          alert("Server has internal error for finding similar clusters");
        }
      );
    },
    taintAndSave: function() {
      const self = this;
      const ret = new Promise((resolve, reject) => {
        const antidfs = root => {
          root.parent = undefined;
          if (
            typeof root.aux !== "undefined" &&
            (root.type === "cluster" || root.type === "trigger") &&
            !root.aux.touched
          ) {
            root.aux.touched = true;
          }
          if (
            typeof root.aux !== "undefined" &&
            root.type === "sentence" &&
            (typeof root.aux.marking === "undefined" ||
              root.aux.marking === constants.Marking.NEUTRAL)
          ) {
            root.aux.touched = true;
            root.aux.marking = constants.Marking.POSITIVE;
          }
          if (
            typeof root !== "undefined" &&
            typeof root.children !== "undefined"
          ) {
            for (let i of root.children) {
              antidfs(i);
              i.parent = undefined;
            }
          }
        };
        antidfs(this.clusters);

        axios({
          baseURL: constants.baseURL,
          url: "/s3/data",
          method: "POST",
          data: { clusters: this.clusters.children },
          params: { session: self.$localStorage.get("session", "dummy") }
        }).then(
          resp => {
            self.clusters.children.map(cluster => {
              cluster.children = cluster.children.map(trigger => {
                trigger.children = trigger.children.map(sentence => {
                  sentence.parent = trigger;
                  return sentence;
                });
                trigger.parent = cluster;
                return trigger;
              });
              cluster.parent = self.clusters;
              return cluster;
            });
            resolve(resp);
            alert("Saved!");
          },
          err => {
            self.clusters.children.map(cluster => {
              cluster.children = cluster.children.map(trigger => {
                trigger.children = trigger.children.map(sentence => {
                  sentence.parent = trigger;
                  return sentence;
                });
                trigger.parent = cluster;
                return trigger;
              });
              cluster.parent = self.clusters;
              return cluster;
            });
            reject(err);
            console.log(err);
            alert("No Saved!unfortunately you can do nothing....");
          }
        );
      });
      return ret;
    },
    saveAndClearBoard: function() {
      const self = this;
      this.taintAndSave().then(success => {
        self.clusters.children.map(cluster => {
          cluster.children.map(trigger => {
            trigger.children = [];
            trigger.aux.blacklist = [];
          });
        });
      });
    },
    saveAndStartTraining: function() {
      const self = this;
	  this.taintAndSave().then(success=>{
		alert("We have started a training process.");
	  });
      
      //  this.taintAndSave().then(success => {
      // 	       axios({
      //     baseURL: constants.baseURL,
      //     url: "/s4/train_model_from_s3",
      //     method: "POST",
      //     params: { session: self.$localStorage.get("session", "dummy") }
      //   }).then(
      //     resp => {
      // 	self.$router.replace({name:"pending_page_for_step_4"});
      //     },
      //     err => {
      //       	console.log(err);
      //     }
      //   );
      //  });
    },
    fetchData: function() {
      const self = this;
      axios({
        baseURL: constants.baseURL,
        url: "/s3/data",
        method: "GET",
        params: { session: self.$localStorage.get("session", "dummy") }
      }).then(
        resp => {
          const root = { key: "root", type: "root", parent: null, aux: {} };
          root.children = resp.data.clusters.map(cluster => {
            cluster.children = cluster.children.map(trigger => {
              trigger.children = trigger.children.map(sentence => {
                sentence.parent = trigger;
                return sentence;
              });
              trigger.parent = cluster;
              if (typeof trigger.aux === "undefined") trigger.aux = {};
              if (typeof trigger.aux.blacklist === "undefined")
                trigger.aux.blacklist = [];
              return trigger;
            });
            if (typeof cluster.aux === "undefined") cluster.aux = {};
            cluster.aux.working = false;
            cluster.parent = root;
            return cluster;
          });
          this.clusters = root;
          self.workingClusterSelectorOn = true;
        },
        err => {
          console.log(err);
        }
      );
    },
    mergeTree: function(src, dst) {
      const self = this;
      Set.prototype.isSuperset = function(subset) {
        for (var elem of subset) {
          if (!this.has(elem)) {
            return false;
          }
        }
        return true;
      };

      Set.prototype.union = function(setB) {
        var union = new Set(this);
        for (var elem of setB) {
          union.add(elem);
        }
        return union;
      };

      Set.prototype.intersection = function(setB) {
        var intersection = new Set();
        for (var elem of setB) {
          if (this.has(elem)) {
            intersection.add(elem);
          }
        }
        return intersection;
      };

      Set.prototype.difference = function(setB) {
        var difference = new Set(this);
        for (var elem of setB) {
          difference.delete(elem);
        }
        return difference;
      };

      const mapingchildFromSrc = {};
      const childFromSrc = new Set();
      if (src.children) {
        src.children.map(item => {
          mapingchildFromSrc[item.key] = item;
          childFromSrc.add(item.key);
        });
      }
      const childFromDst = new Set();
      const mapingchildFromDst = {};
      if (dst.children) {
        dst.children.map(item => {
          mapingchildFromDst[item.key] = item;
          childFromDst.add(item.key);
        });
      } else {
        dst.children = [];
      }
      const shouldTakeCare = childFromSrc.intersection(childFromDst);
      const directMergeFromSrc = childFromSrc.difference(shouldTakeCare);
      for (const item of directMergeFromSrc) {
        mapingchildFromSrc[item].parent = dst;
        dst.children.push(mapingchildFromSrc[item]);
      }
      for (const item of shouldTakeCare) {
        self.mergeTree(mapingchildFromSrc[item], mapingchildFromDst[item]);
        mapingchildFromDst[item].parent = dst;
      }
    },
    clusterNameChangeChecker: function(el, event) {
      const newValue = event.target.value;
      if (true) {
        event.target.value = el.key;
        return;
      }
      // @hqiu: I'm shortcutting this function due to implementation inconsistency in between frontend and backend
      // if (newValue.length < 1) {
      // 	event.target.value = el.key;
      // 	return;
      // }
      // const self = this;
      // const conflictsClusters = [];
      // for (let i = 0; i < el.parent.children.length; i++) {
      // 	if (el.parent.children[i].key === newValue) {
      // 		conflictsClusters.push(el.parent.children[i]);
      // 	}
      // }
      // if (conflictsClusters.length > 1) {
      // 	console.log("This shouldnt happen");
      // 	for (let i = 0; i < conflictsClusters.length; ++i) {
      // 		this.mergeTree(conflictsClusters[i], el);
      // 		const idx = self.clusters.children.indexOf(conflictsClusters[i]);
      // 		self.clusters.children.splice(idx, 1);
      // 	}
      // } else if (conflictsClusters.length === 1) {
      // 	const confirmMerge = confirm(
      // 		"Do you want to merge these two clusters?"
      // 	);
      // 	if (confirmMerge) {
      // 		for (let i = 0; i < conflictsClusters.length; ++i) {
      // 			this.mergeTree(conflictsClusters[i], el);
      // 			const idx = self.clusters.children.indexOf(conflictsClusters[i]);
      // 			self.clusters.children.splice(idx, 1);
      // 		}
      // 		el.key = newValue;
      // 	} else {
      // 		event.target.value = el.key;
      // 	}
      // } else {
      // 	el.key = newValue;
      // }
    },
    popupGroundingPanel: function(cluster) {
      const self = this;
      const curTrigger = cluster["children"].map(item => {
        return item.key;
      });
      axios({
        baseURL: constants.baseURL,
        //url: "/v2/s2/similarcluster",
        url: "/s3/grounding/candidates",
        method: "POST",
        data: { triggers: curTrigger },
        params: { session: self.$localStorage.get("session", "dummy") }
      }).then(
        resp => {
          const candidates = resp.data;
          self.groundingCandidates = candidates;
          self.groundingSelectorOn = true;
          self.selectPendingGroundCluster = cluster;
        },
        err => {
          console.log(err);
          self.selectPendingGroundCluster = null;
        }
      );
    },
    focusClusterNameChangeHandler: function(event) {
      const self = this;
      const newValue = event.target.value;
      let found = false;
      if (newValue.length < 1) {
        self.focusClusterName = "";
        return;
      }
      if (this.clusters.children)
        self.clusters.children.forEach(item => {
          if (item.key === newValue) {
            self.focusClusterName = newValue;
            found |= true;
            return;
          }
        });
      if (found) return;
      const confirmCreateNew = confirm(
        "Do you want to create a new event type?"
      );
      if (confirmCreateNew) {
        self.clusters.children.push({
          key: newValue,
          type: "cluster",
          aux: { touched: false, working: true },
          children: [],
          parent: self.clusters
        });
        self.focusClusterName = newValue;
      } else {
        event.target.value = self.focusClusterName;
      }
    },
    checkMove: function(evt, originalEvent) {
      return true;
    },
    changeEventHandler: function(evt) {},
    addEventHandler: function(evt) {
      const self = this;
      let curItem = evt.item._underlying_vm_;
      const parentFinder = node => {
        let runner = node;
        while (runner) {
          if (runner.root) return runner.root;
          runner = runner.$parent;
        }
        return self.clusters;
      };
      if (typeof curItem === "undefined") {
        let runner = evt.from.__vue__;
        while (runner) {
          if (typeof runner.$vnode.data.key !== "undefined") {
            curItem = this.clusters.children[runner.$vnode.data.key];
            break;
          }
          runner = runner.$parent;
        }
      }

      const curParent = parentFinder(evt.to.__vue__);
      const oldParent = curItem.parent;
      // console.log(curParent);
      // console.log(curItem);

      if (curItem.type === "sentence") {
        if (!oldParent.aux.blacklist) oldParent.aux.blacklist = [];
        oldParent.aux.blacklist.push(curItem.aux.instanceId);
      }

      const DFSMarkSentence = root => {
        const pending_marking = constants.Marking.NEGATIVE;
        if (root.type === "sentence") {
          self
            .markTriggerInstance(root, root.parent.parent.key, pending_marking)
            .then(success => {
              // @hqiu. We should restore the state of sentence to unannotated here since it's under new cluster
              root.aux.touched = false;
              root.aux.marking = constants.Marking.NEUTRAL;
            });
        } else {
          if (root.type === "trigger") {
            for (let i of root.children) {
              DFSMarkSentence(i);
            }
          }
        }
      };
      DFSMarkSentence(curItem);

      const buildPathFromChildToEndPoint = (runner, curItem, endPoint) => {
        const st = [];
        while (runner && runner !== endPoint) {
          const children = [];
          if (st.length === 0) {
            children.push(curItem);
          } else {
            children.push(st[st.length - 1]);
          }
          st.push({
            key: runner.key,
            children: children,
            type: runner.type,
            aux: _.cloneDeep(runner.aux),
            parent: undefined
          });
          if (runner.type === "trigger") {
            st[st.length - 1].aux.blacklist = [];
          }
          runner = runner.parent;
        }
        for (let i = 0; i < st.length - 1; i++) {
          st[i].parent = st[i + 1];
        }
        st.forEach(item => {
          if (item.aux) {
            item.aux.touched = false;
          }
        });

        if (st.length < 1) return st;
        // curItem.parent = st[0];
        st[st.length - 1].parent = endPoint;
        return st;
      };

      const getFirstPossibleClusterNumber = function() {
        const existing = {};
        for (let i = 0; i < self.clusters.children.length; ++i) {
          existing[self.clusters.children[i].key] = true;
        }
        for (let i = 0; i < 1000000; ++i) {
          if (existing["HC" + i]) {
            continue;
          } else {
            return "HC" + i;
          }
        }
      };

      let curCluster = curParent;
      while (curCluster) {
        if (curCluster.type === "cluster") break;
        curCluster = curCluster.parent;
      }
      if (!curCluster) {
        const newPath = buildPathFromChildToEndPoint(
          oldParent,
          curItem,
          self.clusters
        );
        // console.log(newPath);
        curItem.parent = newPath[0];
        newPath[newPath.length - 1].key = getFirstPossibleClusterNumber();
        newPath[newPath.length - 1].aux.working = true;
        self.clusters.children.push(newPath[newPath.length - 1]);
      } else {
        const curIdx = curParent.children.indexOf(curItem);
        curParent.children.splice(curIdx, 1);
        const newPath = buildPathFromChildToEndPoint(
          oldParent,
          curItem,
          self.clusters
        );
        curItem.parent = newPath[0];
        self.mergeTree(newPath[newPath.length - 1], curCluster);
      }

      // Remove the empty parent node after its child was removed
      let runner = oldParent;
      while (runner && runner.children && runner.children.length === 0) {
        const cur = runner;
        runner = runner.parent;
        if (runner !== undefined) {
          const idx = runner.children.indexOf(cur);
          runner.children.splice(idx, 1);
        } else {
          console.log("It should not happen2");
        }
      }
    },
    markTriggerInstance: function(sentence, event_type_name, marking) {
      const self = this;
      console.log("Marking: " + marking);
      const ret = new Promise((resolve, reject) => {
        axios({
          baseURL: constants.baseURL,
          url: "/s3/mark_sentence",
          method: "POST",
          data: {
            trigger_instance: sentence.aux.instanceId,
            event_type_name: event_type_name,
            marking: marking,
            trigger: sentence.parent.aux.trigger
          },
          params: { session: self.$localStorage.get("session", "dummy") }
        }).then(
          resp => {
            sentence.aux.touched = true;
            sentence.aux.marking = marking;
            resolve(resp);
          },
          err => {
            console.log(err);
            reject(err);
          }
        );
      });
      return ret;
    },
    toggleAnnotation: function(el) {
      const self = this;
      let curParent = el.parent;
      let curChild = el;

      const DFSMarkSentence = (root, preference) => {
        if (root.type === "sentence") {
          if (preference === null) {
            if (
              !root.aux.touched ||
              root.aux.marking === constants.Marking.NEUTRAL
            ) {
              const pending_marking = constants.Marking.NEGATIVE;
              self
                .markTriggerInstance(
                  root,
                  root.parent.parent.key,
                  pending_marking
                )
                .then(success => {});
            } else {
              const pending_marking =
                root.aux.marking === constants.Marking.POSITIVE
                  ? constants.Marking.NEGATIVE
                  : constants.Marking.POSITIVE;
              self
                .markTriggerInstance(
                  root,
                  root.parent.parent.key,
                  pending_marking
                )
                .then(success => {});
            }
          } else {
            self
              .markTriggerInstance(root, root.parent.parent.key, preference)
              .then(success => {});
          }
        } else {
          for (let i of root.children) {
            DFSMarkSentence(i, constants.Marking.NEGATIVE);
          }
        }
      };
      DFSMarkSentence(el, null);

      //   if (curChild.type !== "sentence") {
      //     const idx = curParent.children.indexOf(curChild);
      //     curParent.children.splice(idx, 1);
      //   }
    },
    getMoreSentence: function(triggerNode) {
      const self = this;
      const tmpBlackList = Array.concat(
        triggerNode.aux.blacklist,
        triggerNode.children.map(sentence => {
          return sentence.aux.instanceId;
        })
      );
      const triggers = [
        {
          trigger: triggerNode.key,
          postag: null,
          blacklist: tmpBlackList,
          fullTextSearchStr:
            triggerNode.aux.fullTextSearchkey &&
            triggerNode.aux.fullTextSearchkey.length > 1
              ? triggerNode.aux.fullTextSearchkey
              : ""
        }
      ];
      axios({
        baseURL: constants.baseURL,
        url: "/s3/query_unannotated_sentence",
        method: "POST",
        data: { eventType: triggerNode.parent.key, triggers: triggers },
        params: { session: self.$localStorage.get("session", "dummy") }
      }).then(
        resp => {
          const ret = resp.data;
          const touched_root_root = triggerNode.parent;
          const dfs_mark_touch = function(root) {
            root.aux.touched = true;
            if (typeof root.children !== "undefined") {
              for (let i = 0; i < root.children.length; ++i) {
                dfs_mark_touch(root.children[i]);
              }
            }
          };
          dfs_mark_touch(touched_root_root);
          ret.map(trigger => {
            trigger.children.map(sentence => {
              sentence.parent = triggerNode;
              triggerNode.children.push(sentence);
            });
          });
        },
        err => {
          console.log(err);
        }
      );
    },
    importEventTriggers: function(evt) {
      const self = this;
      if (evt.target.files.length < 1) return;
      const file = evt.target.files[0];
      evt.target.value = "";
      const formData = new FormData();
      formData.append("clusters", file);
      axios({
        baseURL: constants.baseURL,
        url: "/v2/s2/4",
        method: "POST",
        data: formData,
        params: { session: self.$localStorage.get("session", "dummy") }
      }).then(
        success => {
          this.$router.replace({ name: success.data.redirect });
        },
        err => {
          console.log(err);
          alert(err);
        }
      );
    },
    submitGroundingDecision: function() {
      const self = this;
      axios({
        baseURL: constants.baseURL,
        url: "/s3/grounding",
        method: "POST",
        params: { session: self.$localStorage.get("session", "dummy") },
        data: {
          currentEventType: self.selectPendingGroundCluster.key,
          groundingCandidate: self.userGroundingDecision
        }
      }).then(
        success => {
          alert(success.data.text);
          if (success.data.success) {
            self.groundingSelectorOn = false;

            const conflictsClusters = [];
            for (let i = 0; i < self.clusters.children.length; i++) {
              if (
                self.clusters.children[i].key ===
                  self.selectPendingGroundCluster.key &&
                self.clusters.children[i] !== self.selectPendingGroundCluster
              ) {
                conflictsClusters.push(self.clusters.children[i]);
              }
            }
            if (conflictsClusters.length === 1) {
              for (let i = 0; i < conflictsClusters.length; ++i) {
                this.mergeTree(conflictsClusters[i], self.selectPendingGroundCluster);
                const idx = self.clusters.children.indexOf(
                  conflictsClusters[i]
                );
                self.clusters.children.splice(idx, 1);
              }
            } else {
              self.selectPendingGroundCluster.key = self.userGroundingDecision;
            }
          }
        },
        fail => {
          console.log(fail);
        }
      );
    },
    selectDisplayCluster: function(mark) {
      const self = this;
      if (this.clusters.children) {
        for (let i = 0; i < this.clusters.children.length; ++i) {
          this.clusters.children[i].aux.working = mark;
        }
      }
    },
    showAnnotatedExample: function(clusterNode) {
      const self = this;
      const triggers = [];
      for (let i = 0; i < clusterNode.children.length; ++i) {
        triggers.push({
          trigger: clusterNode.children[i].aux.trigger,
          postag: clusterNode.children[i].aux.trigger_postag,
          blacklist: []
        });
      }
      axios({
        baseURL: constants.baseURL,
        url: "/s3/get_annotated_sentence",
        method: "POST",
        data: { triggers: triggers, eventType: clusterNode.key },
        params: { session: self.$localStorage.get("session", "dummy") }
      }).then(
        resp => {
          const ret = resp.data;
          ret.map(trigger => {
            trigger.children.map(sentence => {
              sentence.parent = trigger;
            });
            trigger.parent = clusterNode;
          });
          self.mergeTree(
            { key: "dummy", type: "cluster", children: ret },
            clusterNode
          );
        },
        err => {
          console.log(err);
        }
      );
    },
    hideAnnotatedExample: function(clusterNode) {
      const self = this;
      for (let i = 0; i < clusterNode.children.length; ++i) {
        for (let j = clusterNode.children[i].children.length - 1; j >= 0; j--) {
          const sentenceObj = clusterNode.children[i].children[j];
          if (
            sentenceObj.aux.touched &&
            sentenceObj.aux.marking !== constants.Marking.NEUTRAL
          ) {
            clusterNode.children[i].children.splice(j, 1);
          }
        }
      }
    },
    hideCluster: function(clusterPtr) {
      clusterPtr.aux.working = false;
    },
    toggleModalPanel: function(panelSwitch) {
      const self = this;
      self[panelSwitch] = !self[panelSwitch];
    }
  },
  data() {
    return {
      history: 1,
      clusters: {},
      eventhandler: {
        changeEventHandler: this.changeEventHandler,
        addEventHandler: this.addEventHandler,
        toggleAnnotation: this.toggleAnnotation,
        getMoreSentence: this.getMoreSentence,
        toggleAddTriggerPanelOn: this.toggleAddTriggerPanelOn
      },
      newclusters: [],
      focusClusterName: "",
      similarCluster: { type: "root", children: [] },
      similarClusterOn: false,
      similarclusterdisplaystring: "Similar Cluster",
      workingClusterSelectorOn: false,
      addTriggerOn: false,
      triggerLemmaForAddingTrigger: "",
      addTriggerFullTextSearchKey: "",
      addTriggerFocusCluster: null,
      triggerPrefilled: false,
      groundingSelectorOn: false,
      groundingCandidates: [],
      userGroundingHandmakeType: "",
      userGroundingDecision: "",
      addTriggerPanelOn: false,
      selectPendingGroundCluster: null
    };
  },
  components: {
    draggable,
    "local-draggable": local
  },
  mounted() {
    this.fetchData();
    // console.log(this.$route.params);
  },
  computed: {
    focusCluster: function() {
      if (this.clusters.children) {
        for (let i = 0; i < this.clusters.children.length; ++i) {
          if (this.clusters.children[i].key === this.focusClusterName) {
            return this.clusters.children[i];
          }
        }
      }
      return null;
    },
    workingCluster: function() {
      const arr = [];
      if (this.clusters.children) {
        for (let i = 0; i < this.clusters.children.length; ++i) {
          if (this.clusters.children[i].aux.working) {
            arr.push(this.clusters.children[i]);
          }
        }
      }
      return arr;
    }
  }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
#step3 {
  height: 100%;
  overflow-y: hidden;
  display: flex;
  flex-direction: column;
  /* overflow-x: scroll; */
}
#navigate {
  display: flex;
  flex: row;
  height: 3%;
}
#main {
  height: 97%;
  display: flex;
  flex-direction: row;
  /* overflow-x:hidden; */
}
#clusters {
  display: flex;
  flex-direction: row;
  overflow-x: scroll;
}
.touched {
  background-color: aquamarine;
}
.disabledBox[disabled="disabled"]::placeholder {
  color: #000000 !important;
}
</style>
