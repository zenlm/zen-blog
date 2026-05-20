(function(){
  var ns = 'http://www.w3.org/2000/svg';
  function ensoSvg(){
    var svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 512 512');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('width', '22');
    svg.setAttribute('height', '22');
    svg.setAttribute('aria-label', 'Zen LM');
    svg.setAttribute('class', 'zenlm-brand-svg');
    svg.style.cssText = 'display:inline-block;vertical-align:-3px;margin-right:7px;flex-shrink:0';
    var p = document.createElementNS(ns, 'path');
    p.setAttribute('d', 'M 256 84 A 172 172 0 1 0 256 428 A 172 172 0 1 0 256 84 Z');
    p.setAttribute('stroke', 'currentColor');
    p.setAttribute('stroke-width', '36');
    p.setAttribute('stroke-linecap', 'round');
    p.setAttribute('stroke-linejoin', 'round');
    p.setAttribute('class', 'zenlm-brand-path');
    svg.appendChild(p);
    return svg;
  }
  function lucideGithub(){
    var svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '2');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    svg.setAttribute('width', '16');
    svg.setAttribute('height', '16');
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('class', 'zen-link-icon');
    var p = document.createElementNS(ns, 'path');
    p.setAttribute('d', 'M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.4 5.4 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4');
    svg.appendChild(p);
    var p2 = document.createElementNS(ns, 'path');
    p2.setAttribute('d', 'M9 18c-4.51 2-5-2-7-2');
    svg.appendChild(p2);
    return svg;
  }
  function hfEmoji(){
    var s = document.createElement('span');
    s.className = 'zen-link-icon';
    s.style.cssText = 'font-size:16px;line-height:1;margin-right:6px;display:inline-block';
    s.setAttribute('aria-hidden', 'true');
    s.textContent = '🤗';
    return s;
  }
  function lucideFolder(){
    var svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '2');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    svg.setAttribute('width', '14');
    svg.setAttribute('height', '14');
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('class', 'zen-link-icon');
    var p = document.createElementNS(ns, 'path');
    p.setAttribute('d', 'M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z');
    svg.appendChild(p);
    return svg;
  }
  var sectionPaths = {
    '/docs': true,
    '/docs/api': true,
    '/docs/models': true,
    '/docs/datasets': true,
    '/docs/training': true,
  };
  function decorate(){
    document.querySelectorAll('a').forEach(function(a){
      // Strip lotus regardless of decoration state (handles re-renders)
      var walker = document.createTreeWalker(a, NodeFilter.SHOW_TEXT, null);
      var node;
      while ((node = walker.nextNode())) {
        if (node.nodeValue && node.nodeValue.indexOf('🪷') !== -1) {
          node.nodeValue = node.nodeValue.replace(/🪷\s?/g, '');
        }
      }
      var t = (a.textContent || '').trim();
      var href = a.getAttribute('href') || '';
      // The brand link
      if (t === 'Zen LM' && !a.querySelector('svg[aria-label="Zen LM"]')) {
        a.insertBefore(ensoSvg(), a.firstChild);
      }
      // HuggingFace
      else if (/huggingface\.co/i.test(href) && !a.querySelector('.zen-link-icon')) {
        a.insertBefore(hfEmoji(), a.firstChild);
      }
      // GitHub
      else if (/github\.com/i.test(href) && !a.querySelector('.zen-link-icon')) {
        a.insertBefore(lucideGithub(), a.firstChild);
      }
      // Doc section headings
      else if (sectionPaths[href] && !a.querySelector('svg, .zen-link-icon')) {
        a.insertBefore(lucideFolder(), a.firstChild);
      }
    });
  }
  function start(){ decorate(); setInterval(decorate, 800); }
  if (document.readyState !== 'loading') start();
  else document.addEventListener('DOMContentLoaded', start);
})();
