(globalThis.TURBOPACK||(globalThis.TURBOPACK=[])).push(["object"==typeof document?document.currentScript:void 0,21861,e=>{"use strict";let t=(0,e.i(21281).default)("search",[["path",{d:"m21 21-4.34-4.34",key:"14j7rj"}],["circle",{cx:"11",cy:"11",r:"8",key:"4ej97u"}]]);e.s(["Search",()=>t],21861)},9323,21826,56777,e=>{"use strict";var t=e.i(21281);let s=(0,t.default)("airplay",[["path",{d:"M5 17H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2h-1",key:"ns4c3b"}],["path",{d:"m12 15 5 6H7Z",key:"14qnn2"}]]);e.s(["Airplay",()=>s],9323);let n=(0,t.default)("moon",[["path",{d:"M20.985 12.486a9 9 0 1 1-9.473-9.472c.405-.022.617.46.402.803a6 6 0 0 0 8.268 8.268c.344-.215.825-.004.803.401",key:"kfwtm"}]]);e.s(["Moon",()=>n],21826);let a=(0,t.default)("sun",[["circle",{cx:"12",cy:"12",r:"4",key:"4exip2"}],["path",{d:"M12 2v2",key:"tus03m"}],["path",{d:"M12 20v2",key:"1lh1kg"}],["path",{d:"m4.93 4.93 1.41 1.41",key:"149t6j"}],["path",{d:"m17.66 17.66 1.41 1.41",key:"ptbguv"}],["path",{d:"M2 12h2",key:"1t8f8n"}],["path",{d:"M20 12h2",key:"1q8mjw"}],["path",{d:"m6.34 17.66-1.41 1.41",key:"1m8zz5"}],["path",{d:"m19.07 4.93-1.41 1.41",key:"1shlcs"}]]);e.s(["Sun",()=>a],56777)},54359,e=>{"use strict";var t=e.i(52540);function s({size:e=64,color:s="#A855F7",animate:n=!0,loop:a=!1,asLoader:r=!1,className:i=""}){let o=Math.random().toString(36).slice(2,7),l=a?"infinite":"1",c=r?{animation:`zenEnsoSpin-${o} 1.2s linear infinite`}:{};return(0,t.jsxs)("span",{className:`inline-flex items-center justify-center ${i}`,style:{width:e,height:e},"aria-label":"Zen LM",children:[(0,t.jsx)("style",{children:`
        @keyframes zenEnsoDraw-${o} {
          to { stroke-dashoffset: 0; }
        }
        @keyframes zenEnsoFill-${o} {
          0%   { opacity: 0; transform: scale(0.92); filter: blur(6px); }
          60%  { opacity: 1; transform: scale(1.02); filter: blur(1px); }
          100% { opacity: 1; transform: scale(1.00); filter: blur(0px); }
        }
        @keyframes zenEnsoCut-${o} {
          to { opacity: 1; }
        }
        @keyframes zenEnsoSpin-${o} {
          to { transform: rotate(360deg); }
        }
        .zen-enso-stroke-${o} {
          stroke-dasharray: 1200;
          stroke-dashoffset: ${n?"1200":"0"};
          animation: ${n?`zenEnsoDraw-${o} ${a?"1.35s":"1.25s"} cubic-bezier(.2,.9,.25,1) ${l} forwards`:"none"};
        }
        .zen-enso-fill-${o} {
          opacity: ${n?"0":"1"};
          transform-origin: 256px 256px;
          transform: scale(${n?"0.92":"1"});
          animation: ${n?`zenEnsoFill-${o} ${a?"0.60s":"0.55s"} cubic-bezier(.2,.9,.2,1) ${l} forwards`:"none"};
          animation-delay: ${n?"1.05s":"0s"};
        }
        .zen-enso-cut-${o} {
          opacity: ${n?"0":"1"};
          animation: ${n?`zenEnsoCut-${o} 0.01s linear ${l} forwards`:"none"};
          animation-delay: ${n?"1.05s":"0s"};
        }
      `}),(0,t.jsxs)("svg",{viewBox:"0 0 512 512",fill:"none",xmlns:"http://www.w3.org/2000/svg",width:e,height:e,style:c,children:[(0,t.jsx)("circle",{className:`zen-enso-fill-${o}`,cx:"256",cy:"256",r:"172",fill:s}),(0,t.jsx)("path",{className:`zen-enso-stroke-${o}`,d:"M256 74 C154 74 78 160 78 258 C78 352 150 432 250 438 C345 444 434 374 438 276 C441 210 406 150 348 118 C305 94 280 86 256 74",stroke:s,strokeWidth:"26",strokeLinecap:"round",strokeLinejoin:"round"}),(0,t.jsx)("path",{className:`zen-enso-cut-${o}`,d:"M410 148 C392 126 366 106 340 94",stroke:"currentColor",strokeWidth:"34",strokeLinecap:"round"})]})]})}e.s(["ZenEnso",()=>s,"default",0,s])}]);