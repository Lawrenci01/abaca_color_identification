//  DETAIL MODAL
// ============================================================
function openDetail(id,s){
  if(!s||typeof s!=='object')return;
  function safeJson(val,fallback){try{return(typeof val==='string'&&val)?JSON.parse(val):val||fallback;}catch(e){return fallback;}}
  function _safeRgb(rgb){if(!rgb||typeof rgb!=='object')return null;const R=Number(rgb.R),G=Number(rgb.G),B=Number(rgb.B);if(isNaN(R)||isNaN(G)||isNaN(B))return null;return{R,G,B};}
  function _safeLab(lab){if(!lab||typeof lab!=='object')return null;const L=Number(lab.L),a=Number(lab.a),b=Number(lab.b);if(isNaN(L)||isNaN(a)||isNaN(b))return null;return{L,a,b};}
  s.dominant_rgb=_safeRgb(safeJson(s.dominant_rgb_json||s.dominant_rgb,null));
  s.dominant_lab=_safeLab(safeJson(s.dominant_lab_json||s.dominant_lab,null));
  s.top_5=safeJson(s.top_5_json||s.top_5,[]);
  const chip=document.getElementById('dm-status-chip');
  if(s.verified===1){chip.textContent='✅ Verified';chip.style.cssText='background:var(--accent-light);color:var(--accent);border:1px solid var(--accent-lighter);border-radius:7px;font-size:.68rem;font-weight:700;padding:4px 10px;';}
  else if(s.correction){chip.textContent='📝 Corrected';chip.style.cssText='background:#e6f5f2;color:var(--teal);border:1px solid #c0e8e0;border-radius:7px;font-size:.68rem;font-weight:700;padding:4px 10px;';}
  else if(s._unsaved){chip.textContent='📱 Unsaved';chip.style.cssText='background:var(--warn-bg);color:var(--warn);border:1px solid #f0d580;border-radius:7px;font-size:.68rem;font-weight:700;padding:4px 10px;';}
  else{chip.textContent='⏳ Pending';chip.style.cssText='background:var(--card2);color:var(--sub);border:1px solid var(--border);border-radius:7px;font-size:.68rem;font-weight:700;padding:4px 10px;';}
  document.getElementById('dm-title').textContent=s._unsaved?'📱 Unsaved Scan':'Scan Detail';
  document.getElementById('dm-scan-id').textContent=s.batch_id?'Batch: '+s.batch_id:'ID: '+(s.id||id||'—').toString().slice(0,14);
  const heroThumb=document.getElementById('dm-hero-thumb'),heroGrad=document.getElementById('dm-hero-gradient');
  if(s.thumbnail_b64&&s.thumbnail_b64.startsWith('data:')){
    document.getElementById('dm-thumb-img').src=s.thumbnail_b64;document.getElementById('dm-hero-swatch').style.background=s.dominant_hex||'#888';
    document.getElementById('dm-hero-code').textContent=s.rhs_grade||'—';document.getElementById('dm-hero-verdict').textContent=s.verdict||'';
    document.getElementById('dm-hero-de-badge').textContent='ΔE '+(s.delta_e!=null?Number(s.delta_e).toFixed(2):'—');
    heroThumb.style.display='block';heroGrad.style.display='none';
  }else{
    const c1=s.dominant_hex||'#c8ddd0',c2=s.matched_hex||'#9abcaa';
    document.getElementById('dm-nothumb-bg').style.background='linear-gradient(135deg,'+c1+','+c2+')';
    document.getElementById('dm-nh-swatch').style.background=c1;document.getElementById('dm-nh-code').textContent=s.rhs_grade||'—';document.getElementById('dm-nh-de').textContent='ΔE '+(s.delta_e!=null?Number(s.delta_e).toFixed(2):'—');
    heroGrad.style.display='block';heroThumb.style.display='none';
  }
  document.getElementById('dm-gl-grader').textContent=s.grader_name||'—';document.getElementById('dm-gl-grader').className='dm-gl-val'+(s.grader_name?'':' dim');
  document.getElementById('dm-gl-location').textContent=s.location||'—';document.getElementById('dm-gl-location').className='dm-gl-val teal'+(s.location?'':' dim');
  const vc=s.verdict_color||'#1a8c45';
  document.getElementById('dm-verdict-banner').style.cssText='margin:10px 16px 0;padding:14px 16px;border-radius:12px;display:flex;align-items:center;justify-content:space-between;background:'+vc+'14;border:1px solid '+vc+'35;';
  document.getElementById('dm-verdict-lbl').style.color=vc;document.getElementById('dm-verdict').textContent=s.verdict||'—';document.getElementById('dm-verdict').style.color=vc;document.getElementById('dm-verdict-explain').style.color=vc;
  document.getElementById('dm-de').textContent=s.delta_e!=null?Number(s.delta_e).toFixed(2):'—';document.getElementById('dm-de').style.color=vc;
  document.getElementById('dm-sw-scan').style.background=s.dominant_hex||'#ccc';document.getElementById('dm-sw-ref').style.background=s.matched_hex||'#ccc';
  document.getElementById('dm-scan-hex-code').textContent=(s.dominant_hex||'—').toUpperCase();
  const aiCode=s.rhs_grade||'?',corrCode=s.correction?s.correction.toString().trim().toUpperCase():'';
  if(corrCode&&corrCode!==aiCode){document.getElementById('dm-rhs').innerHTML='<span style="text-decoration:line-through;opacity:.5;">'+aiCode+'</span> → <span style="color:var(--teal);">'+corrCode+'</span>';}else{document.getElementById('dm-rhs').textContent=aiCode;}
  document.getElementById('dm-ref-hex').textContent=(s.matched_hex||'—').toUpperCase();
  document.getElementById('dm-score').textContent=s.match_score!=null?Number(s.match_score).toFixed(1)+'%':'—';
  document.getElementById('dm-hex').textContent=(s.dominant_hex||'—').toUpperCase();
  document.getElementById('dm-seg').textContent=s.seg_found!=null?(s.seg_found?'✅ Found':'⚠️ Fallback')+(s.seg_coverage!=null?' ('+Number(s.seg_coverage).toFixed(0)+'%)':''):'—';
  document.getElementById('dm-wb').textContent=s.wb_applied?'✅ Applied':'Not needed';
  const top5=s.top_5||[];const medals=['🥇','🥈','🥉','4','5'];const t5wrap=document.getElementById('dm-top5-wrap');
  if(top5.length>0){document.getElementById('dm-top5-list').innerHTML=top5.map((t,i)=>`<div class="dm-top5-row"><div class="dm-top5-rank" style="color:${i===0?'var(--warn)':'var(--sub)'}">${medals[i]||i+1}</div><div class="dm-top5-sw" style="background:${t.hex||'#888'}"></div><div style="flex:1;"><div class="dm-top5-code">${t.rhs_code}</div><div class="dm-top5-de" style="color:${t.de_color||'#1a8c45'}">ΔE ${t.delta_e} · ${t.de_label||''}</div></div><div class="dm-top5-score" style="color:${i===0?'var(--accent)':'var(--sub)'}">${Number(t.match_score).toFixed(1)}%</div></div>`).join('');t5wrap.style.display='block';}else{t5wrap.style.display='none';}
    const ts = new Date(s.scanned_at).toLocaleString('en-US', {
        weekday: 'short',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
    let statusDisplay = s.verified === 1 ? '✅ Verified' : (corrCode && corrCode !== aiCode ? '📝 Corrected: ' + aiCode + ' → ' + corrCode : '⏳ Pending review');
    document.getElementById('dm-info-table').innerHTML = [
        {k: '📅 Date & Time', v: ts, cls: '', h: false}, {
            k: '🔬 RHS Grade',
            v: aiCode + (corrCode && corrCode !== aiCode ? ' → ✏️ ' + corrCode : ''),
            cls: 'mono',
            h: false
        },
        {k: '👤 Grader', v: s.grader_name || '—', cls: s.grader_name ? 'accent' : '', h: true}, {
            k: '📍 Location',
            v: s.location || '—',
            cls: s.location ? 'teal' : '',
            h: true
        },
        {k: '📦 Batch ID', v: s.batch_id || '—', cls: '', h: false}, s.grader_notes ? {
            k: '📓 Notes',
            v: s.grader_notes,
            cls: '',
            h: false
        } : null,
        {k: '✅ Status', v: statusDisplay, cls: '', h: false},
    ].filter(Boolean).map(r => `<div class="dm-info-row${r.h?' highlight':''}"><span class="dm-info-key">${r.k}</span><span class="dm-info-val ${r.cls||''}">${r.v}</span></div>`).join('');
    const storeKey = s._storeKey || id;
    document.getElementById('dm-actions').innerHTML = s._unsaved
            ? `<button class="btn primary" style="flex:2;" onclick="saveFromDetail('${storeKey}')">💾 Save Scan</button><button class="btn teal" onclick="requestRescanFromDetail()">🔁 Re-Scan</button><button class="btn" onclick="closeDetail()">✕</button>`
            : `<button class="btn teal" style="flex:1;" onclick="requestRescanFromDetail()">🔁 Re-Scan</button><button class="btn primary" style="flex:1;" onclick="closeDetail()">✓ Close</button>`;
    const modal = document.getElementById('detail-modal');
    modal.style.display = 'block';
    modal.classList.add('open');
    modal.scrollTop = 0;
    }

    function requestRescanFromDetail() {
        closeDetail();
        switchScreen('scanner');
    }

    function closeDetail() {
        const modal = document.getElementById('detail-modal');
        modal.style.display = 'none';
        modal.classList.remove('open');
    }

    async function saveFromDetail(storeKey) {
        let entry = _scanStore[storeKey];
        if (!entry || !entry._unsaved) {
            closeDetail();
            return;
        }
        const btn = document.querySelector('#dm-actions .btn.primary');
        if (btn) {
            btn.disabled = true;
            btn.textContent = '💾 Saving...';
        }
        const payload = {
            rhs_grade: entry.rhs_grade,
            user_id: currentUser ? currentUser.id : null,
            rhs_code: entry.rhs_grade,
            delta_e: entry.delta_e,
            match_score: entry.match_score,
            rgb_r: entry.dominant_rgb?.R,
            rgb_g: entry.dominant_rgb?.G,
            rgb_b: entry.dominant_rgb?.B,
            lab_l: entry.dominant_lab?.L,
            lab_a: entry.dominant_lab?.a,
            lab_b: entry.dominant_lab?.b,
            dominant_hex: entry.dominant_hex,
            matched_hex: entry.matched_hex,
            verdict: entry.verdict,
            verdict_color: entry.verdict_color,
            batch_id: entry.batch_id || '',
            grader_notes: entry.grader_notes || '',
            verified: entry.verified || 0,
            correction: entry.correction || '',
            thumbnail_b64: entry.thumbnail_b64 || '',
            location: entry.location || getFullLocation(),
            grader_name: entry.grader_name || (currentUser ? currentUser.username : ''),
        };
        try {
            const r = await fetch('/api/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            const d = await r.json();
            if (d.error) {
                showError('Save failed', d.error);
                if (btn) {
                    btn.disabled = false;
                    btn.textContent = '💾 Save Scan';
                }
                return;
            }
            entry._unsaved = false;
            entry.id = d.scan_id;
            closeDetail();
        } catch (e) {
            showError('Save error', e.message);
            if (btn) {
                btn.disabled = false;
                btn.textContent = '💾 Save Scan';
            }
        }
    }

    // ============================================================