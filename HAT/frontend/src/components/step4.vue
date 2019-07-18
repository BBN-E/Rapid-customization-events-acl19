<template>
  <div class="step4">
    <div style="display:flex">
      <button @click="TouchAllInstance(true)" style="flex:1">Mark All As annotated</button>
      <button @click="TouchAllInstance(false)" style="flex:1">Mark All As unannotated</button>
      <button @click="SaveAnnotation()" style="flex:1">Save</button>
      <button @click="RetrainModel()" style="flex:1">Retrain The Model</button>
    </div>
    <div style="display:flex">
      <select @change="changeEventType($event)" :value="currentEventType">
        <option
          v-for="(option,idx) in possibleEventTypes"
          :key="idx"
          :value="option.value"
        >{{option.text}}</option>
      </select>
      <select @change="changeArgumentType($event)" :value="currentArgumentType">
        <option
          v-for="(option,idx) in possibleArgumentTypes"
          :key="idx"
          :value="option.value"
        >{{option.text}}</option>
      </select>
      <select @change="changePage($event)" :value="currentPage">
        <option
          v-for="(option,idx) in possiblePages"
          :key="idx"
          :value="option.value"
        >{{option.text}}</option>
      </select>
    </div>
    <ul class="ui list clusters" style="text-align:left;">
      <li
        class="sentences"
        :class="{'sentenceword':idx === taggingsentenceidx}"
        v-for="(sentence,idx) in sentences"
        :key="idx"
        @click="mouseSelectWord(idx,0)"
      >
        <span @click="markSentence(sentence,idx)" :class="{'touched':!sentence.aux.touched}">X</span>
        <span>&nbsp;</span>
        <span
          v-for="(word,idx2) in sentence.aux.token_array"
          :key="idx2"
          :class="{'markbad': sentence.aux.marking === Marking.NEGATIVE}"
        >
          <!-- <span :class="mappingCSS(idx,idx2)">{{word}}</span> -->
          <span
            :class="{'triggerlemma':sentence.aux.tags.trigger.indexOf(idx2) !== -1,'special':sentence.aux.tags.trigger.indexOf(idx2) === -1 && sentence.aux.tags[sentence.aux.argumentType].indexOf(idx2) !== -1}"
          >{{word}}</span>
          <span>&nbsp;</span>
        </span>
      </li>
    </ul>
  </div>
</template>

<script>
import constants from "@/constants.js";
import axios from "axios";
export default {
  name: "step4",
  created: function() {
    window.addEventListener("keypress", this.keyboardHandler);
  },
  beforeDestroy: function() {
    window.removeEventListener("keypress", this.keyboardHandler);
  },
  data() {
	  const self = this;
    return {
      taggingsentenceidx: -1,
      taggingwordidx: -1,
      history: [],
      sentences: [],
      metadata: {},
      currentEventType: "",
      currentArgumentType: "",
      currentPage: -1,
	  Marking:constants.Marking
    };
  },
  watch: {},
  computed: {
    possibleEventTypes: function() {
      const self = this;
      const keySet = Object.keys(self.metadata);
      const ret = [];
      for (let i of keySet) {
        ret.push({ text: i, value: i });
      }
      return ret;
    },
    possibleArgumentTypes: function() {
      const self = this;
      if (self.currentEventType === "") {
        return [];
      }
      const keySet = Object.keys(self.metadata[self.currentEventType]);
      const ret = [];
      for (let i of keySet) {
        ret.push({ text: i, value: i });
      }
      return ret;
    },
    possiblePages: function() {
      const self = this;
      const ret = [];
      if (self.currentEventType === "" || self.currentArgumentType === "") {
        return [];
      }
      for (
        let i = 0;
        i <= self.metadata[self.currentEventType][self.currentArgumentType] ||
        0;
        ++i
      ) {
        ret.push({ text: i, value: i });
      }
      return ret;
    }
  },
  mounted() {
    const self = this;
    axios({
      baseURL: constants.baseURL,
      url: "/s4/metadata",
      method: "GET",
      params: { session: self.$localStorage.get("session", "dummy") }
    }).then(
      resp => {
        const en = Object.entries(resp.data)[0];
        self.currentEventType = en[0];
        const en2 = Object.entries(resp.data[self.currentEventType])[0];
        self.currentArgumentType = en2[0];
        self.currentPage = 0;
        self.metadata = resp.data;
        self.LoadAnnotation(
          self.currentEventType,
          self.currentArgumentType,
          self.currentPage
        );
      },
      err => {
        alert("Server side error");
        console.log(err);
      }
    );
  },
  methods: {
    TouchAllInstance: function(touched) {
      const self = this;
      for (const i of self.sentences) {
        i["aux"]["touched"] = touched;
      }
    },
    SaveAnnotation: function() {
      const self = this;
      const ret = new Promise((resolve, reject) => {
        axios({
          baseURL: constants.baseURL,
          url: "/s4/page",
          method: "POST",
          params: {
            session: self.$localStorage.get("session", "dummy"),
            eventType: self.currentEventType,
            argumentType: self.currentArgumentType,
            page: self.currentPage
          },
          data: { sentences: this.sentences }
        }).then(
          resp => {
            resolve(resp);
          },
          err => {
            alert("Server side error");
            reject(err);
          }
        );
      });
      return ret;
    },
    LoadAnnotation: function(eventType, argumentType, Page) {
      const self = this;
      axios({
        baseURL: constants.baseURL,
        url: "/s4/page",
        method: "GET",
        params: {
          session: self.$localStorage.get("session", "dummy"),
          eventType: eventType,
          argumentType: argumentType,
          page: Page
        }
      }).then(resp => {
        self.sentences = resp.data;
        self.currentEventType = eventType;
        self.currentArgumentType = argumentType;
        self.currentPage = Page;
      });
    },
    toggleTagging: function(enabled) {
      if (!enabled) {
        window.removeEventListener("keypress", this.keyboardHandler);
      } else {
        window.addEventListener("keypress", this.keyboardHandler);
      }
    },
    markSentence: function(sentence, idx) {
      if (sentence.aux.marking === constants.Marking.POSITIVE) {
        sentence.aux.marking = constants.Marking.NEGATIVE;
      } else {
        sentence.aux.marking = constants.Marking.POSITIVE;
      }
    },
    jumpToCluster: function() {
      this.toggleTagging(false);

      this.toggleTagging(true);
    },
    keyboardHandler: function(event) {
      const key = event.key;
      if (this.taggingsentenceidx === -1 || this.taggingwordidx === -1) {
        this.taggingsentenceidx = 0;
        this.taggingwordidx = 0;
        // return;
      }
      let targetSentence;
      let changeLine;
      switch (key) {
        case "w":
        case "W":
          targetSentence = Math.max(this.taggingsentenceidx - 1, 0);
          this.taggingwordidx = Math.min(
            this.taggingwordidx,
            this.sentences[targetSentence].aux.token_array.length - 1
          );
          this.taggingsentenceidx = targetSentence;
          break;
        case "s":
        case "S":
          targetSentence = Math.min(
            this.taggingsentenceidx + 1,
            this.sentences.length - 1
          );
          this.taggingwordidx = Math.min(
            this.taggingwordidx,
            this.sentences[targetSentence].aux.token_array.length - 1
          );
          this.taggingsentenceidx = targetSentence;
          break;
        case "a":
        case "A":
          if (this.taggingwordidx === 0 && this.taggingsentenceidx === 0) break;
          changeLine = this.taggingwordidx === 0;
          this.taggingsentenceidx = changeLine
            ? Math.max(this.taggingsentenceidx - 1, 0)
            : this.taggingsentenceidx;
          this.taggingwordidx = changeLine
            ? this.sentences[this.taggingsentenceidx].aux.token_array.length - 1
            : this.taggingwordidx - 1;
          break;
        case "d":
        case "D":
          if (
            this.taggingsentenceidx === this.sentences.length - 1 &&
            this.taggingwordidx ===
              this.sentences[this.sentences.length - 1].aux.token_array.length -
                1
          )
            break;
          changeLine =
            this.taggingwordidx ===
            this.sentences[this.taggingsentenceidx].aux.token_array.length - 1;
          this.taggingsentenceidx = changeLine
            ? Math.min(this.taggingsentenceidx + 1, this.sentences.length - 1)
            : this.taggingsentenceidx;
          this.taggingwordidx = changeLine ? 0 : this.taggingwordidx + 1;
          break;
        case "1":
          if (this.taggingsentenceidx === -1 || this.taggingwordidx === -1)
            return;
          // this.clearWord("time", this.taggingsentenceidx, this.taggingwordidx);
          // this.toggleWord("location", this.taggingsentenceidx, this.taggingwordidx);
          break;
        case "2":
          if (this.taggingsentenceidx === -1 || this.taggingwordidx === -1)
            return;
          // this.clearWord("location", this.taggingsentenceidx, this.taggingwordidx);
          // this.toggleWord("time", this.taggingsentenceidx, this.taggingwordidx);
          break;
        case "3":
          this.markSentence(
            this.sentences[this.taggingsentenceidx],
            this.taggingsentenceidx
          );
          break;
        case "4":
          this.undo();
          break;
        case "[":
          this.currentPage = parseInt(this.currentPage);
          if (this.currentPage > 0) {
            this.SaveAnnotation().then(success => {
              this.LoadAnnotation(this.currentType, this.currentPage - 1);
            });
          }
          break;
        case "]":
          this.currentPage = parseInt(this.currentPage);
          if (this.currentPage < this.metadata[this.currentType]) {
            this.SaveAnnotation().then(success => {
              this.LoadAnnotation(this.currentType, this.currentPage + 1);
            });
          }
          break;
        case "j":
        case "J":
          this.jumpToCluster();
          break;
      }
    },
    mouseSelectWord: function(sentenceidx, wordidx) {
      this.taggingsentenceidx = sentenceidx;
      this.taggingwordidx = wordidx;
    },
    toggleWord: function(word_type, sentenceidx, wordidx) {},
    undo: function() {
      if (this.history.length < 1) return;
    },
    changeEventType: function(evt) {
      const self = this;
      const oldVal = this.currentEventType;
      const newVal = evt.target.value;
      self.SaveAnnotation().then(
        success => {
          self.LoadAnnotation(
            newVal,
            Object.entries(self.metadata[newVal])[0][0],
            0
          );
        },
        fail => {
          evt.target.value = oldVal;
        }
      );
    },
    changeArgumentType: function(evt) {
      const self = this;
      const oldVal = this.currentArgumentType;
      const newVal = evt.target.value;
      self.SaveAnnotation().then(
        success => {
          self.LoadAnnotation(self.currentEventType, newVal, 0);
        },
        fail => {
          evt.target.value = oldVal;
        }
      );
    },
    changePage: function(evt) {
      const self = this;
      const oldVal = this.currentPage;
      const newVal = evt.target.value;
      self.SaveAnnotation().then(success => {
        self.LoadAnnotation(
          self.currentEventType,
          self.currentArgumentType,
          newVal
        );
      });
    },
    RetrainModel: function() {
    const self = this;
    alert("We have started a training process.");
    //   axios({
    //     baseURL: constants.baseURL,
    //     url: "/s4/train_model_from_s4",
    //     method: "POST",
    //     params: { session: self.$localStorage.get("session", "dummy") }
    //   }).then(
    //     resp => {
    //       this.$router.replace({ name: "pending_page_for_step_4" });
    //     },
    //     err => {
    //       alert("Server side error");
    //     }
    //   );
    }
  }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.sentences {
  /* float: left; */
  margin-top: 5px;
}
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
.sentenceword {
  background-color: rgb(222, 232, 252);
}
.triggerlemma {
  color: red;
  text-decoration: underline;
  font-weight: bold;
}
.special {
  color: spe;
  text-decoration: underline;
  font-weight: bold;
}
.clusters {
  flex: 50;
  overflow-y: auto;
}
.step4 {
  display: flex;
  flex-direction: column;
  height: 100%;
  margin: 0;
}
.touched {
  background-color: aquamarine;
}
.markbad {
  text-decoration-color: red;
  text-decoration-line: line-through;
  text-decoration-style: solid;
}
</style>
