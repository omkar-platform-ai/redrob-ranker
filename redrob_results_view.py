"""Redesigned results view for the Redrob Ranker sandbox.

DROP-IN — replaces the evidence-ledger / honeypot / JD-keyword render block in
``scripts/demo_app.py``. It is PURE PRESENTATION: it reads only fields that the
ranking engine already put on each ``ScoredCandidate`` (the five sub-scores, the
composite ``score``, ``rank``, ``reasoning``, ``matched_skills``,
``hidden_gem_reasons``, ``is_honeypot``, title/company/yoe/location/notice) plus
the honeypot-reason strings you already re-derive demo-side.

It calls NOTHING in ``src/``. No scoring, no embedding, no LLM, no network. The
master/detail interaction runs client-side inside a single
``streamlit.components.v1.html`` iframe, so selecting a candidate never triggers a
Streamlit rerun.

Wire-up in scripts/demo_app.py — replace the whole "Results: the evidence ledger"
block (everything under ``else:`` after ``results = st.session_state.get(...)``)
with:

    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from redrob_results_view import render_results_view

    render_results_view(
        ranked=results["ranked"],
        parsed_jd=results["parsed_jd"],
        names=results["names"],
        honeypots=results["honeypots"],   # list of {id,name,title,company,reasons}
        csv_data=results["csv"],
    )

Nothing above that block (hero, input card, pipeline, session_state) changes.
"""

from __future__ import annotations

import json
import streamlit.components.v1 as components

# Mirrors src/config.WEIGHTS — kept as a literal so this render layer imports
# nothing from src/. (label, color, weight) in fusion-weight order.
_SIGNALS = [
    ("semantic",   "Semantic",   "#6366f1", 0.40),
    ("role_fit",   "Role-fit",   "#0ea5e9", 0.20),
    ("skill",      "Skill",      "#10b981", 0.15),
    ("behavioral", "Behavioral", "#f59e0b", 0.15),
    ("career",     "Career",     "#8b5cf6", 0.10),
]

_GEM_LABELS = {"open_source": "Open-source contributor", "multi_promotion": "2+ promotions"}


def _humanize_gem(reason: str) -> str:
    return _GEM_LABELS.get(reason, reason.replace("_", " "))


def _g(sc, attr, default=0.0):
    return getattr(sc, attr, default)


def _candidate_payload(sc, names: dict) -> dict:
    """Flatten one ScoredCandidate into the JSON the client renderer consumes."""
    gems = list(_g(sc, "hidden_gem_reasons", []) or [])
    try:
        yoe = float(_g(sc, "years_of_experience", 0) or 0)
    except (TypeError, ValueError):
        yoe = 0.0
    return {
        "id": sc.candidate_id,
        "name": names.get(sc.candidate_id, sc.candidate_id),
        "title": _g(sc, "current_title", "") or "—",
        "company": _g(sc, "current_company", "") or "",
        "yoe": round(yoe, 1),
        "loc": _g(sc, "location", "") or "",
        "notice": _g(sc, "notice_period_days", None),
        "flagged": bool(_g(sc, "is_honeypot", False)),
        "score": float(_g(sc, "score", 0.0) or 0.0),
        "rank": int(_g(sc, "rank", 0) or 0),
        "s": {
            "semantic":   float(_g(sc, "semantic_score", 0.0) or 0.0),
            "role_fit":   float(_g(sc, "role_fit_score", 0.0) or 0.0),
            "skill":      float(_g(sc, "skill_score", 0.0) or 0.0),
            "behavioral": float(_g(sc, "behavioral_score", 0.0) or 0.0),
            "career":     float(_g(sc, "career_score", 0.0) or 0.0),
        },
        "skills": list(_g(sc, "matched_skills", []) or []),
        "gem": _humanize_gem(gems[0]) if gems else None,
        "why": _g(sc, "reasoning", "") or "",
    }


def render_results_view(ranked, parsed_jd, names, honeypots, csv_data, *, height: int = 820) -> None:
    """Render the redesigned results screen as a self-contained component."""
    cands = [_candidate_payload(sc, names) for sc in ranked]
    hp_reasons = {hp["id"]: hp.get("reasons", []) for hp in (honeypots or [])}

    jd = {
        "role": getattr(parsed_jd, "title", "") or "Role",
        "seniority": (getattr(parsed_jd, "seniority_level", "") or "").title() or "—",
        "band": "{}\u2013{} yrs".format(
            getattr(parsed_jd, "min_experience_years", "?"),
            getattr(parsed_jd, "max_experience_years", "?"),
        ),
        "required": list(getattr(parsed_jd, "required_skills", []) or [])[:8],
        "flags": list(getattr(parsed_jd, "disqualifiers", []) or [])[:8],
    }

    payload = {"cands": cands, "honeypotReasons": hp_reasons, "jd": jd, "csv": csv_data or ""}
    payload_json = json.dumps(payload).replace("</", "<\\/")
    html = _TEMPLATE.replace("__PAYLOAD__", payload_json)
    components.html(html, height=height, scrolling=True)


_TEMPLATE = r"""
<!doctype html><html><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{height:100%}
  body{font-family:'Instrument Sans',system-ui,sans-serif;color:#0f172a;-webkit-font-smoothing:antialiased;background:#f6f7f9}
  .mono{font-family:'JetBrains Mono',monospace}
  ::-webkit-scrollbar{width:9px;height:9px}
  ::-webkit-scrollbar-thumb{background:#cfd4de;border-radius:9px;border:2px solid transparent;background-clip:padding-box}
  .app{height:100vh;display:flex;flex-direction:column;background:#f6f7f9;overflow:hidden}
  .hdr{flex:0 0 auto;z-index:20;background:#f6f7f9}
  .topbar{flex:0 0 auto;display:flex;align-items:center;gap:18px;padding:0 22px;height:58px;background:#fff;border-bottom:1px solid #e7e9ef}
  .logo{width:11px;height:11px;border-radius:3px;background:#4f46e5;transform:rotate(45deg)}
  .pill{font-size:11px;font-weight:600;color:#6b7280;background:#f1f2f6;padding:3px 9px;border-radius:6px;letter-spacing:.02em}
  .vr{width:1px;height:22px;background:#e7e9ef}
  .btn{display:inline-flex;align-items:center;gap:7px;height:34px;padding:0 14px;border:1px solid #4f46e5;background:#4f46e5;color:#fff;font-family:inherit;font-size:13px;font-weight:600;border-radius:8px;cursor:pointer}
  .btn:hover{background:#4338ca;border-color:#4338ca}
  .controls{flex:0 0 auto;display:flex;align-items:center;gap:16px;padding:9px 22px;background:#fbfbfc;border-bottom:1px solid #eceef3}
  .ghost{display:inline-flex;align-items:center;gap:7px;height:30px;padding:0 11px;border:1px solid #e2e4ec;background:#fff;color:#475569;font-family:inherit;font-size:12px;font-weight:600;border-radius:7px;cursor:pointer}
  .ghost:hover{border-color:#cfd3df}
  .seg{display:inline-flex;background:#eef0f4;border-radius:8px;padding:2px}
  .seg button{height:26px;padding:0 11px;border:none;background:transparent;color:#64748b;font-family:inherit;font-size:12px;font-weight:600;border-radius:6px;cursor:pointer}
  .seg button.on{background:#fff;color:#4f46e5;box-shadow:0 1px 2px rgba(15,23,42,.12)}
  .lbl{font-size:11px;font-weight:600;color:#64748b;letter-spacing:.03em}
  .jdpanel{flex:0 0 auto;padding:14px 22px;background:#f3f1fb;border-bottom:1px solid #e6e2f6;display:flex;flex-wrap:wrap;gap:26px;align-items:flex-start}
  .jdk{font-size:11px;font-weight:700;color:#6354a6;letter-spacing:.06em}
  .chip{font-size:12px;font-weight:500;padding:3px 9px;border-radius:999px;white-space:nowrap}
  .chip-req{background:#fff;color:#4338ca;border:1px solid #ddd6fb}
  .chip-flag{background:#fdeef0;color:#a8324b;border:1px solid #f6d6dd}
  .main{flex:1;min-height:0;display:flex}
  .rail{flex:0 0 440px;display:flex;flex-direction:column;background:#fff;border-right:1px solid #e7e9ef;min-height:0}
  .tabs{flex:0 0 auto;display:flex;gap:4px;padding:12px 14px 0}
  .tabs button{flex:1;height:34px;border:none;background:#f7f8fa;color:#475569;font-family:inherit;font-size:13px;font-weight:600;border-radius:8px 8px 0 0;cursor:pointer;border-bottom:2px solid transparent}
  .tabs button.on{background:#fff;color:#0f172a;border-bottom-color:#4f46e5}
  .tabs button.on.flag{color:#b42318;border-bottom-color:#b42318}
  .list{flex:1;min-height:0;overflow-y:auto;padding:6px 10px 16px}
  .row{display:flex;gap:11px;align-items:center;padding:11px 10px;margin-top:4px;border-radius:10px;cursor:pointer;background:#fff;border:1px solid transparent}
  .row:hover{background:#f7f8fa}
  .row.sel{background:#f1f0fc;border-color:#d7d3f7}
  .row.compact{padding:8px 9px}
  .rk{flex:0 0 auto;width:26px;text-align:center;font-size:13px;font-weight:700;color:#475569}
  .rk.top{color:#4f46e5}
  .rname{font-size:14px;font-weight:600;color:#0f172a;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .rsub{font-size:12px;color:#64748b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .mini{display:flex;height:5px;margin-top:7px;background:#eef0f5;border-radius:999px;overflow:hidden}
  .gemdot{width:6px;height:6px;border-radius:999px;background:#d4a017;display:inline-block}
  .badge{font-size:16px;font-weight:700;line-height:1}
  .badge-l{font-size:10px;font-weight:700;letter-spacing:.04em;text-transform:uppercase}
  .detail{flex:1;min-width:0;overflow-y:auto}
  .dwrap{max-width:1100px;margin:0;padding:30px 52px 80px}
  .dwrap.compact{padding:20px 44px 60px}
  .dhead{display:flex;justify-content:space-between;align-items:flex-start;gap:20px}
  .dname{font-size:23px;font-weight:700;letter-spacing:-.02em}
  .gembig{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:700;color:#8a5a06;background:#fdf3da;border:1px solid #f0dca0;padding:3px 9px;border-radius:999px}
  .dmeta{font-size:13px;color:#64748b;margin-top:6px}
  .score-big{font-size:40px;font-weight:700;line-height:.9}
  .sect-t{font-size:13px;font-weight:700;letter-spacing:-.01em}
  .barbig{display:flex;height:30px;margin-top:11px;background:#f0f1f5;border-radius:9px;overflow:hidden;box-shadow:inset 0 0 0 1px rgba(15,23,42,.04)}
  .brow{display:flex;align-items:center;gap:12px;padding:9px 10px;border-bottom:1px solid #f1f2f6;border-radius:8px}
  .brow.compact{padding:6px 10px}
  .brow.clickable{cursor:pointer;transition:background .12s,box-shadow .12s}
  .brow.clickable:hover{background:#e7e8ff;box-shadow:inset 3px 0 0 #4f46e5}
  .brow .chev{flex:0 0 16px;text-align:center;color:#aab0c0;font-size:12px;font-weight:700;transition:transform .15s,color .15s}
  .brow.clickable:hover .chev{color:#4f46e5}
  .brow.open{background:#f0f0fe}
  .brow.open .chev{transform:rotate(90deg);color:#4f46e5}
  .evhint{font-size:11px;font-weight:600;color:#4f46e5;background:#eef0fe;border:1px solid #dcd9fb;padding:3px 9px;border-radius:999px;white-space:nowrap}
  .bdot{width:10px;height:10px;border-radius:3px}
  .corro{margin-top:22px;display:flex;align-items:center;gap:14px;padding:13px 16px;background:#f7f8fa;border:1px solid #ecedf2;border-radius:11px}
  .cdot{width:12px;height:12px;border-radius:999px}
  .kicker{font-size:12px;font-weight:700;color:#6b7280;letter-spacing:.02em}
  .sk{font-size:12px;font-weight:500;background:#ecfdf3;color:#15803d;border:1px solid #cdeed8;padding:4px 10px;border-radius:999px;white-space:nowrap}
  .why{font-size:14px;line-height:1.6;color:#334155;margin-top:8px;text-wrap:pretty}
  .hp{margin-top:22px;background:#fdf1f2;border:1px solid #f6d2d8;border-radius:12px;padding:16px 18px}
  .micro{font-size:11px;color:#64748b}
  .railtools{flex:0 0 auto;padding:8px 12px 6px;display:flex;flex-direction:column;gap:8px;border-bottom:1px solid #f1f2f6}
  .search{width:100%;height:34px;padding:0 12px;border:1px solid #e2e4ec;border-radius:9px;font-family:inherit;font-size:13px;color:#0f172a;background:#f7f8fa;outline:none}
  .search:focus{border-color:#c7c3f3;box-shadow:0 0 0 3px rgba(79,70,229,.10)}
  .filters{display:flex;flex-wrap:wrap;gap:6px}
  .fchip{font-size:11.5px;font-weight:600;padding:4px 10px;border-radius:999px;border:1px solid #e2e4ec;background:#fff;color:#64748b;cursor:pointer;white-space:nowrap}
  .fchip:hover{border-color:#cfd3df}
  .fchip.on{background:#eef0fe;border-color:#c9c2f7;color:#4338ca}
  .delta{display:inline-flex;align-items:center;font-family:'JetBrains Mono',monospace;font-size:10.5px;font-weight:700;padding:1px 6px;border-radius:6px;white-space:nowrap}
  .delta-up{color:#15803d;background:#ecfdf3}
  .delta-dn{color:#b42318;background:#fdeef0}
  .delta-eq{color:#475569;background:#f1f2f6}
  .avail{display:inline-flex;align-items:center;gap:5px;font-size:11.5px;font-weight:600;padding:3px 9px;border-radius:999px;white-space:nowrap}
  .availdot{width:6px;height:6px;border-radius:999px;display:inline-block}
  .emptyrail{padding:30px 16px;text-align:center;color:#64748b;font-size:13px}
  .sigev{font-size:12px;color:#475569;background:#f7f8fa;border-radius:8px;padding:9px 12px;margin:2px 0 8px;line-height:1.5}
  .sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
  [role="option"]:focus-visible,[role="button"]:focus-visible,button:focus-visible,.fchip:focus-visible,.search:focus-visible{outline:2px solid #4f46e5;outline-offset:2px}
  @media (max-width:820px){.app{height:auto;overflow:visible}.main{flex-direction:column}.rail{flex:0 0 auto;border-right:none;border-bottom:1px solid #e7e9ef}.detail{overflow:visible}}
  @media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head>
<body><div class="app" id="app"></div><div id="srlive" aria-live="polite" aria-atomic="true" class="sr-only"></div>
<script>
var D = __PAYLOAD__;
var SIG = [["semantic","Semantic","#6366f1",.40],["role_fit","Role-fit","#0ea5e9",.20],["skill","Skill","#10b981",.15],["behavioral","Behavioral","#f59e0b",.15],["career","Career","#8b5cf6",.10]];
var ranked = D.cands.filter(function(c){return !c.flagged;});
var flagged = D.cands.filter(function(c){return c.flagged;});
var st = { sel:(ranked[0]||D.cands[0]||{}).id, mode:"fit", density:"comfortable", tab:"ranked", jd:true, rankMode:"multi", q:"", sigOpen:{}, filters:{gem:false,inband:false,india:false,notice:false,active:false} };
// Baseline ranks: engine (multi-signal) vs keyword-only (semantic similarity).
(function(){ var m=ranked.slice().sort(function(a,b){return b.score-a.score;}); m.forEach(function(c,i){c._mr=i+1;}); var k=ranked.slice().sort(function(a,b){return (b.s.semantic||0)-(a.s.semantic||0);}); k.forEach(function(c,i){c._kr=i+1;}); })();
var BAND=(String(D.jd.band||"").match(/\d+/g)||[]).map(Number);
var ABROAD=/(usa|u\.s|uk|canada|australia|singapore|uae|dubai|toronto|new york|sydney|london|berlin|germany|netherlands|ireland)/i;
var SCORES=ranked.map(function(c){return c.score;}); var MINS=SCORES.length?Math.min.apply(null,SCORES):0, MAXS=SCORES.length?Math.max.apply(null,SCORES):1;

function esc(s){return String(s==null?"":s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;");}
function tier(c){ if(c.flagged) return {l:"Zeroed",col:"#b42318"}; var f=c.score*100; if(f>=70)return{l:"Strong fit",col:"#4f46e5"}; if(f>=50)return{l:"Possible",col:"#b45309"}; return{l:"Weak fit",col:"#64748b"}; }
function grade(c){ var f=c.score*100; return f>=75?"A":f>=65?"B":f>=55?"C":f>=45?"D":"E"; }
function badge(c){ if(c.flagged) return st.mode==="raw"?"0.000":st.mode==="grade"?"\u2014":"0"; if(st.mode==="raw")return c.score.toFixed(2); if(st.mode==="grade")return grade(c); return ""+Math.round(c.score*100); }
function segs(c){ return SIG.map(function(s){ var sub=(c.s[s[0]]||0), contrib=sub*s[3]; return {key:s[0],label:s[1],color:s[2],weight:s[3],sub:sub,contrib:contrib,widthPct:(contrib*100).toFixed(2),subPct:(sub*100).toFixed(1)}; }); }
function fusion(c){ return SIG.reduce(function(a,s){return a+(c.s[s[0]]||0)*s[3];},0); }
function corro(c){ var dots=SIG.map(function(s){return (c.s[s[0]]||0)>=0.5;}); var n=dots.filter(Boolean).length; var lab=n>=4?"Strong corroboration":n===3?"Moderate corroboration":"Mixed signals"; var col=n>=4?"#15803d":n===3?"#b45309":"#64748b"; return {dots:dots,n:n,lab:lab,col:col}; }
function miniBar(c){ return '<div class="mini" aria-hidden="true">'+segs(c).map(function(s){return '<div style="width:'+s.widthPct+'%;background:'+s.color+'"></div>';}).join("")+'</div>'; }
function avail(c){
  var b=c.s.behavioral||0, n=c.notice;
  if(b<0.4) return {l:"Low engagement",col:"#6b7280",bg:"#f1f2f6",bd:"#e2e4ec"};
  if(n!=null && n<=30) return {l:"Available · "+n+"d notice",col:"#15803d",bg:"#ecfdf3",bd:"#cdeed8"};
  if(b>=0.6) return {l:"Active candidate",col:"#15803d",bg:"#ecfdf3",bd:"#cdeed8"};
  return {l:"Open to roles",col:"#b45309",bg:"#fdf4e7",bd:"#f3e0c0"};
}
function deltaInfo(c){ var d=(c._kr||0)-(c._mr||0); if(d>0) return {cls:"delta-up",txt:"▲"+d}; if(d<0) return {cls:"delta-dn",txt:"▼"+(-d)}; return {cls:"delta-eq",txt:"–"}; }
function passFilters(c){
  var q=st.q.trim().toLowerCase();
  if(q){ var hay=(c.name+" "+c.title+" "+c.company+" "+(c.skills||[]).join(" ")).toLowerCase(); if(hay.indexOf(q)<0) return false; }
  var f=st.filters;
  if(f.gem && !c.gem) return false;
  if(f.inband && BAND.length===2 && !(c.yoe>=BAND[0] && c.yoe<=BAND[1])) return false;
  if(f.india && ABROAD.test(c.loc||"")) return false;
  if(f.notice && !(c.notice!=null && c.notice<=30)) return false;
  if(f.active && !((c.s.behavioral||0)>=0.6)) return false;
  return true;
}
function displayList(){
  var base = st.tab==="ranked"?ranked.slice():flagged.slice();
  if(st.tab==="ranked"){ base.sort(function(a,b){ return st.rankMode==="keyword"?(a._kr-b._kr):(a._mr-b._mr); }); base=base.filter(passFilters); }
  return base;
}
function sigEvidence(c,key){
  if(key==="semantic") return "Embedding cosine similarity between the profile and the JD — captures implicit fit the keyword list misses.";
  if(key==="role_fit") return "Title + company-type + location ("+(c.loc||"—")+") + experience band (this candidate: "+c.yoe+"yr vs "+(D.jd.band||"")+").";
  if(key==="skill") return (c.skills&&c.skills.length)?("Proficiency-weighted match on required skills — matched: "+c.skills.slice(0,6).join(", ")+"."):"No required skills matched in this profile.";
  if(key==="behavioral") return "Recency + engagement + recruiter-response + notice ("+(c.notice!=null?c.notice+"d":"n/a")+"). Availability: "+avail(c).l+".";
  if(key==="career") return c.gem?("Trajectory, stability & hidden-gem signal — "+c.gem+"."):"Trajectory, stability & progression velocity across the career history.";
  return "";
}

function rowHTML(c){
  var t=tier(c), dense=st.density==="compact";
  var dr=st.tab==="ranked"?(st.rankMode==="keyword"?c._kr:c._mr):c.rank;
  var di=deltaInfo(c), av=avail(c), selected=c.id===st.sel;
  var arialbl="Rank "+dr+", "+c.name+(c.title?", "+c.title:"")+", "+t.l+(c.flagged?", honeypot, zeroed":", score "+Math.round(c.score*100)+" of 100");
  return '<div class="row'+(selected?" sel":"")+(dense?" compact":"")+'" data-id="'+esc(c.id)+'" role="option" aria-selected="'+(selected?"true":"false")+'" tabindex="'+(selected?"0":"-1")+'" aria-label="'+esc(arialbl)+'">'
    +'<div class="rk mono'+(dr<=3?" top":"")+'" aria-hidden="true">#'+dr+'</div>'
    +'<div style="flex:1;min-width:0">'
      +'<div style="display:flex;align-items:center;gap:6px"><span class="rname">'+esc(c.name)+'</span>'+(c.gem?'<span class="gemdot" title="Hidden gem" aria-hidden="true"></span>':'')+(c.flagged?'':'<span class="availdot" title="'+esc(av.l)+'" aria-hidden="true" style="background:'+av.col+'"></span>')+(c.flagged?'':'<span class="delta '+di.cls+'" title="engine vs keyword-only rank" aria-hidden="true">'+di.txt+'</span>')+'</div>'
      +'<div class="rsub">'+esc(c.title)+(c.company?" \u00b7 "+esc(c.company):"")+'</div>'
      +miniBar(c)
    +'</div>'
    +'<div style="display:flex;flex-direction:column;align-items:flex-end;gap:2px">'
      +'<span class="badge mono" style="color:'+t.col+'">'+badge(c)+'</span>'
      +'<span class="badge-l" style="color:'+t.col+'">'+t.l+'</span>'
    +'</div></div>';
}

function detailHTML(c){
  if(!c) return "";
  var t=tier(c), dense=st.density==="compact";
  var av=c.flagged?null:avail(c);
  var dr2=c.flagged?c.rank:(st.rankMode==="keyword"?c._kr:c._mr);
  var fit=Math.round(c.score*100), g=c.flagged?"\u2014":grade(c);
  var big,sub;
  if(c.flagged){ big="0"; sub="0.0000 composite"; }
  else if(st.mode==="raw"){ big=c.score.toFixed(3); sub=fit+" / 100 fit"; }
  else if(st.mode==="grade"){ big=g; sub=fit+" \u00b7 "+c.score.toFixed(3); }
  else { big=""+fit; sub=c.score.toFixed(4)+" composite"; }

  var meta=esc(c.title)+" @ "+esc(c.company)+"  \u00b7  "+c.yoe+"yr  \u00b7  "+esc(c.loc)+(c.notice!=null?"  \u00b7  "+c.notice+"d notice":"");
  var h='<div class="dwrap'+(dense?" compact":"")+'">';

  h+='<div class="dhead"><div style="min-width:0">'
    +'<div style="display:flex;align-items:center;gap:10px"><span class="mono" style="font-size:13px;font-weight:700;color:#475569">#'+dr2+'</span><span class="dname">'+esc(c.name)+'</span>'
    +(c.gem?'<span class="gembig">\u25c6 '+esc(c.gem)+'</span>':'')+'</div>'
    +'<div class="dmeta mono">'+meta+'</div>'
    +(c.flagged?'':'<div style="display:flex;align-items:center;gap:10px;margin-top:9px;flex-wrap:wrap"><span class="avail" style="color:'+av.col+';background:'+av.bg+';border:1px solid '+av.bd+'"><span class="availdot" style="background:'+av.col+'"></span>'+av.l+'</span><span class="mono" style="font-size:11.5px;color:#64748b">multi-signal #'+c._mr+' · keyword-only would rank #'+c._kr+'</span></div>')
    +'</div>'
    +'<div style="text-align:right;flex:0 0 auto"><div class="score-big mono" style="color:'+t.col+'">'+big+'</div>'
    +'<div style="font-size:12px;font-weight:600;color:'+t.col+';margin-top:5px;text-transform:uppercase;letter-spacing:.04em">'+t.l+'</div>'
    +'<div class="micro mono" style="margin-top:3px">'+sub+'</div>'
    +(c.flagged?'':'<div class="micro mono" style="margin-top:3px">#'+c._mr+' of '+ranked.length+' · top '+Math.max(1,Math.round((c._mr/(ranked.length||1))*100))+'%</div><div style="width:130px;height:6px;background:#eef0f5;border-radius:999px;margin:8px 0 0 auto;position:relative"><span style="position:absolute;top:-1.5px;left:'+(MAXS>MINS?((c.score-MINS)/(MAXS-MINS)*100):50).toFixed(1)+'%;transform:translateX(-50%);width:9px;height:9px;border-radius:999px;background:'+t.col+';box-shadow:0 0 0 2px #fff,0 0 0 3px '+t.col+'"></span></div>')
    +'</div></div>';

  if(c.flagged){
    var reasons=(D.honeypotReasons[c.id]||[]);
    h+='<div class="hp" role="region" aria-label="Honeypot detail"><div style="font-size:13px;font-weight:700;color:#b42318" role="heading" aria-level="3">\u26d3 Honeypot \u2014 composite forced to 0</div>'
      +'<div style="font-size:13px;color:#9b3b3b;margin-top:6px;line-height:1.5">Tripped 2+ impossible-profile signals, so it can\'t game the ranking. Over the full 100K pool it falls below the top-100 cutoff entirely.</div>'
      +'<div style="display:flex;flex-direction:column;gap:7px;margin-top:13px">'
      +reasons.map(function(r){return '<div style="display:flex;gap:9px;align-items:flex-start;font-size:13px;color:#7f1d1d"><span class="mono" style="color:#d97179;margin-top:1px">\u2715</span><span>'+esc(r)+'</span></div>';}).join("")
      +'</div></div>';
  } else {
    var S=segs(c), fus=fusion(c), adj=c.score-fus;
    h+='<div style="margin-top:26px"><div style="display:flex;align-items:baseline;justify-content:space-between"><span class="sect-t" role="heading" aria-level="3">Score composition</span><span class="evhint">\u25b8 click any signal for the evidence behind it</span></div>';
    h+='<div class="barbig" aria-hidden="true">'+S.map(function(s){return '<div style="width:'+s.widthPct+'%;background:'+s.color+'" title="'+s.label+': '+s.contrib.toFixed(3)+'  ('+s.sub.toFixed(2)+' \u00d7 '+s.weight.toFixed(2)+')"></div>';}).join("")+'</div>';
    h+='<div style="margin-top:16px">';
    h+=S.map(function(s){
      var sopen=!!st.sigOpen[s.key];
      var siglbl=s.label+" signal, score "+s.sub.toFixed(2)+" of 1, contributes "+s.contrib.toFixed(3)+" to composite. Activate to show evidence.";
      return '<div class="brow clickable'+(dense?" compact":"")+(sopen?" open":"")+'" data-sig="'+s.key+'" role="button" tabindex="0" aria-expanded="'+(sopen?"true":"false")+'" aria-label="'+esc(siglbl)+'">'
        +'<div style="flex:0 0 132px;display:flex;align-items:center;gap:9px"><span class="bdot" aria-hidden="true" style="background:'+s.color+'"></span><span style="font-size:13px;font-weight:600;color:#1e293b">'+s.label+'</span><span class="mono" style="font-size:10px;font-weight:700;color:#64748b">'+Math.round(s.weight*100)+'%</span></div>'
        +'<div style="flex:1;height:7px;background:#eef0f5;border-radius:999px;overflow:hidden" aria-hidden="true"><div style="height:100%;width:'+s.subPct+'%;background:'+s.color+';opacity:.92"></div></div>'
        +'<div class="mono" style="flex:0 0 48px;text-align:right;font-size:13px;font-weight:600;color:#334155">'+s.sub.toFixed(2)+'</div>'
        +'<div class="mono" style="flex:0 0 66px;text-align:right;font-size:12px;color:#64748b">+'+s.contrib.toFixed(3)+'</div>'
        +'<span class="chev" aria-hidden="true">\u25b8</span></div>'
        +(sopen?'<div class="sigev">'+esc(sigEvidence(c,s.key))+'</div>':'');
    }).join("");
    if(Math.abs(adj)>0.005){
      h+='<div class="brow"><div style="flex:0 0 132px;font-size:13px;font-weight:600;color:#a8516b">Multipliers</div>'
        +'<div style="flex:1;font-size:12px;color:#64748b">role-fit / location / career caps</div>'
        +'<div style="flex:0 0 48px"></div>'
        +'<div class="mono" style="flex:0 0 66px;text-align:right;font-size:12px;color:'+(adj<0?"#b42318":"#15803d")+'">'+(adj<0?"":"+")+adj.toFixed(3)+'</div><span class="chev" style="visibility:hidden">\u25b8</span></div>';
    }
    h+='<div style="display:flex;align-items:center;gap:12px;padding:9px 0 0"><div style="flex:0 0 132px;font-size:13px;font-weight:700">Composite</div><div style="flex:1"></div><div style="flex:0 0 48px"></div><div class="mono" style="flex:0 0 66px;text-align:right;font-size:13px;font-weight:700">'+c.score.toFixed(4)+'</div><span style="flex:0 0 14px"></span></div></div></div>';

    var cr=corro(c);
    h+='<div class="corro"><div style="display:flex;gap:5px" aria-hidden="true">'+cr.dots.map(function(on){return '<span class="cdot" style="background:'+(on?cr.col:"#d6dae2")+'"></span>';}).join("")+'</div>'
      +'<div><span style="font-size:13px;font-weight:700;color:'+cr.col+'">'+cr.lab+'</span><span style="font-size:13px;color:#64748b"> \u2014 '+cr.n+' of 5 signals above 0.50</span></div></div>';

    if(c.skills&&c.skills.length){
      h+='<div style="margin-top:22px"><span class="kicker" role="heading" aria-level="3">MATCHED SKILLS</span><div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:9px">'
        +c.skills.map(function(s){return '<span class="sk">'+esc(s)+'</span>';}).join("")+'</div></div>';
    }
  }

  h+='<div style="margin-top:22px"><span class="kicker" role="heading" aria-level="3">WHY THIS RANK</span><p class="why">'+esc(c.why||"Disqualified \u2014 flagged as a honeypot and forced to score 0.")+'</p>'
    +'<div class="micro" style="margin-top:10px">Template-generated \u00b7 no LLM at rank time \u00b7 non-hallucinated from profile fields</div></div>';
  h+='</div>';
  return h;
}

function topFit(){ var c=ranked[0]; if(!c) return "\u2014"; return st.mode==="raw"?c.score.toFixed(2):st.mode==="grade"?grade(c):""+Math.round(c.score*100); }

function render(){
  var list = displayList();
  var selC = D.cands.filter(function(c){return c.id===st.sel;})[0]||ranked[0];
  var jd=D.jd;

  var html='';
  html+='<div class="hdr">';
  html+='<div class="topbar"><div style="display:flex;align-items:center;gap:10px"><span style="font-size:12px;font-weight:700;color:#6b7280;letter-spacing:.05em;text-transform:uppercase">Evidence ledger</span></div>'
    +'<div class="vr"></div>'
    +'<div style="display:flex;align-items:center;gap:8px;min-width:0;flex:1"><span style="font-size:13px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'+esc(jd.role)+'</span><span class="mono" style="font-size:12px;color:#64748b;white-space:nowrap">'+esc(jd.band)+'</span></div>'
    +'<div style="display:flex;align-items:center;gap:14px;flex:0 0 auto"><div class="mono" style="display:flex;gap:16px;font-size:12px;color:#6b7280">'
      +'<span><b style="color:#111827">'+ranked.length+'</b> ranked</span><span><b style="color:#b42318">'+flagged.length+'</b> zeroed</span><span>top <b style="color:#4f46e5">'+topFit()+'</b></span></div>'
    +'<button class="btn" data-act="csv"><span class="mono" style="font-size:14px">\u2193</span> CSV</button></div></div>';

  html+='<div class="controls"><button data-act="jd" style="display:inline-flex;align-items:center;gap:8px;height:32px;padding:0 14px;border:1px solid #c9c2f7;background:#eceafe;color:#4338ca;font-family:inherit;font-size:12.5px;font-weight:700;border-radius:8px;cursor:pointer"><span style="font-size:13px">'+(st.jd?"\u25be":"\u25b8")+'</span> '+(st.jd?"Hide":"Show")+' what the engine understood</button><div style="flex:1"></div>'
    +'<div style="display:flex;align-items:center;gap:7px"><span class="lbl">RANK BY</span><div class="seg">'
      +'<button data-rank="multi" class="'+(st.rankMode==="multi"?"on":"")+'">Multi-signal</button>'
      +'<button data-rank="keyword" class="'+(st.rankMode==="keyword"?"on":"")+'">Keyword-only</button></div></div>'
    +'<div style="display:flex;align-items:center;gap:7px"><span class="lbl">SCORE</span><div class="seg">'
      +'<button data-mode="fit" class="'+(st.mode==="fit"?"on":"")+'">Fit /100</button>'
      +'<button data-mode="raw" class="mono '+(st.mode==="raw"?"on":"")+'">0\u20131</button>'
      +'<button data-mode="grade" class="'+(st.mode==="grade"?"on":"")+'">Grade</button></div></div>'
    +'<div style="display:flex;align-items:center;gap:7px"><span class="lbl">DENSITY</span><div class="seg">'
      +'<button data-dens="comfortable" class="'+(st.density==="comfortable"?"on":"")+'">Comfortable</button>'
      +'<button data-dens="compact" class="'+(st.density==="compact"?"on":"")+'">Compact</button></div></div></div>';
  html+='</div>';

  if(st.jd){
    html+='<div class="jdpanel" role="region" aria-label="What the engine understood from the JD">'
      +'<div style="display:flex;flex-direction:column;gap:2px"><span class="jdk">ROLE</span><span style="font-size:13px;font-weight:600">'+esc(jd.role)+'</span></div>'
      +'<div style="display:flex;flex-direction:column;gap:2px"><span class="jdk">SENIORITY</span><span style="font-size:13px;font-weight:600">'+esc(jd.seniority)+'</span></div>'
      +'<div style="display:flex;flex-direction:column;gap:2px"><span class="jdk">EXPERIENCE</span><span class="mono" style="font-size:13px;font-weight:600">'+esc(jd.band)+'</span></div>'
      +'<div style="flex:1;min-width:240px;display:flex;flex-direction:column;gap:5px"><span class="jdk">REQUIRED \u2014 KEYWORD EXTRACTION, NO LLM</span><div style="display:flex;flex-wrap:wrap;gap:5px">'+(jd.required||[]).map(function(k){return '<span class="chip chip-req">'+esc(k)+'</span>';}).join("")+'</div></div>'
      +(jd.flags&&jd.flags.length?'<div style="display:flex;flex-direction:column;gap:5px"><span class="jdk" style="color:#a8516b">JD-SPECIFIC FLAGS</span><div style="display:flex;flex-wrap:wrap;gap:5px;max-width:280px">'+jd.flags.map(function(f){return '<span class="chip chip-flag">'+esc(f)+'</span>';}).join("")+'</div></div>':'')
      +'</div>';
  }

  var railTools = st.tab==="ranked" ? ('<div class="railtools"><input class="search" data-search placeholder="Search name, title, skill…" value="'+esc(st.q)+'"><div class="filters">'+[["gem","Hidden gems"],["active","Available now"],["inband","In-band YoE"],["india","India-based"],["notice","≤30d notice"]].map(function(p){return '<button class="fchip'+(st.filters[p[0]]?" on":"")+'" data-filter="'+p[0]+'">'+p[1]+'</button>';}).join("")+'</div></div>') : '';
  var rowsHtml = list.length ? list.map(rowHTML).join("") : '<div class="emptyrail">No candidates match these filters.</div>';
  html+='<div class="main"><div class="rail"><div class="tabs">'
    +'<button class="'+(st.tab==="ranked"?"on":"")+'" data-tab="ranked">Ranked · '+ranked.length+'</button>'
    +'<button class="'+(st.tab==="flagged"?"on flag":"")+'" data-tab="flagged">Honeypots · '+flagged.length+'</button></div>'
    +railTools
    +'<div class="list" role="listbox" aria-label="Ranked candidates — use arrow keys to navigate, Enter to open">'+rowsHtml+'</div></div>'
    +'<div class="detail" role="region" aria-label="Candidate detail">'+detailHTML(selC)+'</div></div>';

  document.getElementById("app").innerHTML=html;

  var lr=document.getElementById("srlive");
  if(lr&&selC){
    var lrk=selC.flagged?selC.rank:(st.rankMode==="keyword"?selC._kr:selC._mr);
    lr.textContent="Showing "+selC.name+", rank "+lrk+" of "+ranked.length+", "+tier(selC).l+(selC.flagged?", honeypot, zeroed":", "+Math.round(selC.score*100)+" of 100 fit");
  }
}

var ACT_SEL="[data-id],[data-act],[data-mode],[data-dens],[data-tab],[data-rank],[data-filter],[data-sig]";
function activate(el){
  if(el.dataset.id){ st.sel=el.dataset.id; render(); }
  else if(el.dataset.act==="csv"){ var b=new Blob([D.csv],{type:"text/csv"}); var a=document.createElement("a"); a.href=URL.createObjectURL(b); a.download="submission.csv"; document.body.appendChild(a); a.click(); a.remove(); }
  else if(el.dataset.act==="jd"){ st.jd=!st.jd; render(); }
  else if(el.dataset.mode){ st.mode=el.dataset.mode; render(); }
  else if(el.dataset.dens){ st.density=el.dataset.dens; render(); }
  else if(el.dataset.tab){ st.tab=el.dataset.tab; var L=st.tab==="ranked"?ranked:flagged; if(L.length && !L.some(function(c){return c.id===st.sel;})) st.sel=L[0].id; render(); }
  else if(el.dataset.rank){ st.rankMode=el.dataset.rank; render(); }
  else if(el.dataset.filter){ st.filters[el.dataset.filter]=!st.filters[el.dataset.filter]; render(); }
  else if(el.dataset.sig){ st.sigOpen[el.dataset.sig]=!st.sigOpen[el.dataset.sig]; render(); }
}
document.getElementById("app").addEventListener("click",function(e){
  var el=e.target.closest(ACT_SEL); if(!el) return; activate(el);
});
// Keyboard activation for the non-native controls (rows/signal rows are role-divs).
// Native <button>/<input> self-activate on Enter/Space, so skip them here.
document.getElementById("app").addEventListener("keydown",function(e){
  if(e.key!=="Enter"&&e.key!==" "&&e.key!=="Spacebar") return;
  var el=e.target.closest(ACT_SEL); if(!el || el.tagName==="BUTTON" || el.tagName==="INPUT") return;
  e.preventDefault();
  var sig=el.dataset.sig, isRow=!!el.dataset.id;
  activate(el);
  if(sig){ var s2=document.querySelector('[data-sig="'+sig+'"]'); if(s2) s2.focus(); }
  else if(isRow){ var r2=document.querySelector('.row[aria-selected="true"]'); if(r2) r2.focus(); }
});
document.getElementById("app").addEventListener("input",function(e){
  if(e.target && e.target.matches && e.target.matches("[data-search]")){ st.q=e.target.value; render(); var inp=document.querySelector("[data-search]"); if(inp){ inp.focus(); try{inp.selectionStart=inp.selectionEnd=inp.value.length;}catch(_){} } }
});
window.addEventListener("keydown",function(e){
  if(e.key!=="ArrowDown"&&e.key!=="ArrowUp") return;
  if(document.activeElement&&document.activeElement.matches&&document.activeElement.matches("[data-search]")) return;
  var list=displayList(); var i=list.findIndex(function(c){return c.id===st.sel;}); if(i<0) return;
  e.preventDefault(); st.sel=list[e.key==="ArrowDown"?Math.min(list.length-1,i+1):Math.max(0,i-1)].id; render();
  var r=document.querySelector('.row[aria-selected="true"]'); if(r) r.focus();
});
render();
</script></body></html>
"""
