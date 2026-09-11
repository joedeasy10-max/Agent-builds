// Decision-rule matcher for the candidate review page.
//
// The RULES TABLE itself is not here: it is exported from
// src/candidate_review.py (DECISION_RULES) and embedded into the page at build
// time, so thresholds, ordering and reasons exist in exactly one place. This
// file is only the ~20-line interpreter — a transliteration of _rule_matches()
// and decide() in that module.
//
// tests/test_rules_parity.py runs this file against the Python implementation
// over every combination of criteria, confidence, duplicate, negative and
// has-passage, and fails on any disagreement. That test is the reason this
// duplication is safe: the two implementations cannot drift silently, which is
// the bug class that has already cost this project four fixes.

function makeDecider(RULES){
  function ruleMatches(when, facts){
    for(var key in when){
      var want = when[key];
      if(key === "any_fail_in"){
        var any = want.some(function(c){ return facts.fails[c]; });
        if(!any) return false;
      } else if(key === "all_pass"){
        var allPass = RULES.criteria.every(function(c){
          return facts.criteria[c] === "pass";
        }) && !facts.duplicate;
        if(allPass !== want) return false;
      } else if(key === "min_confidence"){
        if(facts.confidence < want) return false;
      } else if(["complete","is_negative","has_passage","duplicate"].indexOf(key) > -1){
        if(!!facts[key] !== !!want) return false;
      } else {
        return false;   // unknown condition: never match rather than guess
      }
    }
    return true;
  }

  return function decide(criteria, confidence, duplicate, isNegative, hasPassage){
    var complete = RULES.criteria.every(function(c){
      return criteria[c] === "pass" || criteria[c] === "fail";
    });
    var fails = {};
    RULES.criteria.forEach(function(c){ fails[c] = criteria[c] === "fail"; });
    var facts = {complete:complete, is_negative:!!isNegative, has_passage:!!hasPassage,
                 duplicate:!!duplicate, confidence:+confidence,
                 criteria:criteria, fails:fails};
    for(var i=0;i<RULES.rules.length;i++){
      var r = RULES.rules[i];
      if(ruleMatches(r.when, facts)){
        return {decision:r.decision, reason:r.reason, rule:r.id};
      }
    }
    return {decision:"review", reason:"No rule matched.", rule:"fallback"};
  };
}

if(typeof module !== "undefined" && module.exports){ module.exports = {makeDecider}; }
