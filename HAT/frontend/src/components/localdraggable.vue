<template>
	<draggable :element="'div'" :list="root.children" class="dragArea" :options="{group:{ name:'general'}}" :move="movecallback" @change="eventhandlers.changeEventHandler" @add="eventhandlers.addEventHandler">
		<sui-accordion v-for="(el,idx) in root.children" :key="idx" style="text-align:left">
			<sui-accordion-title>
				<div class="triggerlemma">
					<i class="minus icon" :class="touched(el)" @click="eventhandlers.toggleAnnotation(el)"></i>
					<p style="word-wrap:break-word;" :class="{'trigger':el.type === 'trigger','sentencebrief': el.type==='sentence'}">{{el.aux && el.aux.abstract?el.aux.abstract:el.key}}</p>
					<div v-if="el.type === 'trigger'">
						<!-- <button @click="eventhandlers.toggleAddTriggerPanelOn(el)">A</button> -->
						<button @click="eventhandlers.getMoreSentence(el)">More</button>
					</div>
				</div>
			</sui-accordion-title>
			<sui-accordion-content :class="{'active': true}">
				<div v-if="el.type === 'sentence'" :class="annotated(el)">
					<span v-for="(word,idx2) in el.aux.token_array" :key="idx2">
						<span :class="HighlightTrigger(el.aux,idx2)">{{word}}</span>
						<span>&nbsp;</span>
					</span>
				</div>
				<local-draggable :root="el" :movecallback="movecallback" :eventhandlers="eventhandlers"></local-draggable>
			</sui-accordion-content>
		</sui-accordion>
	</draggable>
</template>

<script>
import draggable from "vuedraggable";
import constants from "@/constants.js";
export default {
	name: "local-draggable",
	data() {
		return {};
	},
	props: ["root", "movecallback", "eventhandlers"],
	components: {
		draggable
	},
	mounted() {
		// console.log(this.eventhandlers);
	},
	methods: {
		HighlightTrigger: function (sentence, wordidx) {
			return {
				triggerWordInSentence: sentence.tags.trigger.indexOf(wordidx) !== -1
			};
		},
		touched: function (el) {
			return {
				touched: el.aux && !el.aux.touched
			};
		},
		annotated: function(el){
			const positive = (el.aux.marking === constants.Marking.POSITIVE);
			const negative = (el.aux.marking === constants.Marking.NEGATIVE);
			return {
				sentenceArea:true,
				annotated: el.aux && el.aux.touched,
				// @hqiu: We only present positive examples to user.
				markGood: el.aux && el.aux.touched && positive,
				markBad: el.aux && el.aux.touched && negative
			}
		}
	}
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.dragArea {
  /* min-height: 35px; */
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  height: 100%;
}
.triggerWordInSentence {
  color: red;
}
.triggerlemma {
  display: flex;
  flex-direction: row;
  justify-content: space-between;
}
.touched {
  background-color: aquamarine;
}
.trigger {
  color: red;
  /* display: block; */
  width: 50%;
  text-align: center;
}
.sentencebrief {
  color: #467cd3;
}
.markGood {
	background-color: #ccffcc
}
.markBad{
	background-color: #ffcccc
}
</style>
