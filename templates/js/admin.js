//  ADMIN
    // ============================================================
    function switchAdminScreen(name) {
        document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        const sc = document.getElementById('screen-admin-' + name);
        if (sc) sc.classList.add('active');
        const ni = document.getElementById('nav-admin-' + name);
        if (ni) ni.classList.add('active');
        if (name === 'dashboard') admLoadDashboard();
        if (name === 'history') {
            admLoadScans().then(admApplyFilters);
        }
        if (name === 'users') admLoadUsers();
        if (name === 'settings') admLoadSettings();
    }

    async function admLoadDashboard() {
        const [scans, users] = await Promise.all([fetch('/api/admin/scans').then(r => r.json()).catch(() => []), fetch('/api/admin/users').then(r => r.json()).catch(() => [])]);
        admAllScans = scans;
        admAllUsers = users;
        const today = new Date().toDateString();
        const totalToday = scans.filter(s => new Date(s.scanned_at).toDateString() === today).length;
        const verified = scans.filter(s => s.verified === 1);
        const avgDe = scans.length ? (scans.reduce((a, s) => a + (parseFloat(s.delta_e) || 0), 0) / scans.length).toFixed(1) : '—';
        const gradeCounts = {};
        scans.forEach(s => {
            if (s.rhs_grade) {
                gradeCounts[s.rhs_grade] = (gradeCounts[s.rhs_grade] || 0) + 1;
            }
        });
        const sortedGrades = Object.entries(gradeCounts).sort((a, b) => b[1] - a[1]);
        const topGrade = sortedGrades[0] ? sortedGrades[0][0] : '—';
        const maxCount = sortedGrades[0] ? sortedGrades[0][1] : 1;

        // Active users in last 7 days
        const week = Date.now() - 7 * 24 * 60 * 60 * 1000;
        const activeUsers = new Set(scans.filter(s => new Date(s.scanned_at).getTime() > week).map(s => s.user_id || s.username)).size;

        document.getElementById('adm-total').textContent = scans.length;
        document.getElementById('adm-today-sub').textContent = totalToday + ' today';
        document.getElementById('adm-users').textContent = users.length;
        document.getElementById('adm-active-sub').textContent = activeUsers + ' active';
        document.getElementById('adm-verified').textContent = verified.length;
        document.getElementById('adm-verified-pct').textContent = scans.length ? Math.round(verified.length / scans.length * 100) + '%' : '0%';
        document.getElementById('adm-avgde').textContent = avgDe;
        document.getElementById('adm-top-grade').textContent = 'top: ' + topGrade;

        // Grade distribution
        const gradeDist = document.getElementById('adm-grade-dist');
        if (sortedGrades.length === 0) {
            gradeDist.innerHTML = '<div class="admin-empty">No scan data yet.</div>';
        } else {
            gradeDist.innerHTML = sortedGrades.slice(0, 8).map(([code, cnt]) => `
      <div class="grade-bar-row">
        <div class="grade-bar-code">${code}</div>
        <div class="grade-bar-track"><div class="grade-bar-fill" style="width:${Math.round(cnt/maxCount*100)}%"></div></div>
        <div class="grade-bar-count">${cnt}</div>
      </div>`).join('');
        }

        // Activity feed — latest 8 scans
        const feed = document.getElementById('adm-activity-feed');
        const latest = scans.slice(0, 8);
        if (latest.length === 0) {
            feed.innerHTML = '<div class="admin-empty">No activity yet.</div>';
        } else {
            feed.innerHTML = latest.map(s => {
                const ts = new Date(s.scanned_at);
                const timeStr = ts.toLocaleTimeString('en-US', {hour: '2-digit', minute: '2-digit'});
                const dateStr = ts.toLocaleDateString('en-US', {month: 'short', day: 'numeric'});
                const isOk = (s.delta_e != null && parseFloat(s.delta_e) < 5);
                return `<div class="activity-item">
        <div class="activity-dot ${isOk?'ok':'warn'}"></div>
        <div class="activity-swatch" style="background:${s.dominant_hex||'#ccc'}"></div>
        <div class="activity-body">
          <div class="activity-title">${s.rhs_grade||'?'} · ΔE ${s.delta_e!=null?parseFloat(s.delta_e).toFixed(1):'—'} · ${s.username||s.grader_name||'—'}</div>
          <div class="activity-sub">${s.location||'No location'} · ${s.verdict||'—'}</div>
        </div>
        <div class="activity-time">${dateStr}<br>${timeStr}</div>
      </div>`;
            }).join('');
        }
    }

    // ============================================================
    //  ADMIN HISTORY
    // ============================================================
    async function admLoadScans() {
        try {
            const r = await fetch('/api/admin/scans');
            admAllScans = await r.json();
            // Build filter dropdowns
            const graders = [...new Set(admAllScans.map(s => s.username || s.grader_name).filter(Boolean))].sort();
            const locs = [...new Set(admAllScans.map(s => s.location).filter(Boolean))].sort();
            const rhsCodes = [...new Set(admAllScans.map(s => s.rhs_grade).filter(Boolean))].sort();
            const ddG = document.getElementById('adm-dd-grader');
            ddG.innerHTML = '<div class="admin-dd-item" onclick="admSetGrader(\'\')">All Graders</div>' + graders.map(g => `<div class="admin-dd-item" onclick="admSetGrader('${g}')">${g}</div>`).join('');
            const ddL = document.getElementById('adm-dd-loc');
            ddL.innerHTML = '<div class="admin-dd-item" onclick="admSetLoc(\'\')">All Locations</div>' + locs.map(l => `<div class="admin-dd-item" onclick="admSetLoc('${l}')">${l}</div>`).join('');
            const ddR = document.getElementById('adm-dd-rhs');
            ddR.innerHTML = '<div class="admin-dd-item" onclick="admSetRhs(\'\')">All Grades</div>' + rhsCodes.map(c => `<div class="admin-dd-item" onclick="admSetRhs('${c}')">${c}</div>`).join('');
        } catch (e) {
            admAllScans = [];
        }
    }

    let admOpenDD = null;

    function admToggleDD(which) {
        if (admOpenDD && admOpenDD !== which) document.getElementById('adm-dd-' + admOpenDD)?.classList.remove('open');
        const dd = document.getElementById('adm-dd-' + which);
        if (!dd) return;
        if (admOpenDD === which) {
            dd.classList.remove('open');
            admOpenDD = null;
        } else {
            dd.classList.add('open');
            admOpenDD = which;
        }
    }

    document.addEventListener('click', e => {
        if (!e.target.closest('.admin-dropdown-wrap')) {
            document.querySelectorAll('.admin-dropdown').forEach(d => d.classList.remove('open'));
            admOpenDD = null;
        }
    });

    function admSearchHistory(v) {
        admHistSearch = v;
        admApplyFilters();
    }

    function admSetStatus(v) {
        admFilterStatus = v;
        ['all', 'verified', 'pending'].forEach(k => {
            document.getElementById('adm-fp-' + k)?.classList.toggle('active', k === v);
        });
        admApplyFilters();
    }

    function admSetGrader(v) {
        admFilterGrader = v;
        const fp = document.getElementById('adm-fp-grader');
        if (fp) {
            fp.className = 'admin-fp' + (v ? ' active' : '');
            fp.textContent = v ? '👤 ' + v + ' ▾' : '👤 Grader ▾';
        }
        document.getElementById('adm-dd-grader')?.classList.remove('open');
        admOpenDD = null;
        admApplyFilters();
    }

    function admSetLoc(v) {
        admFilterLoc = v;
        const fp = document.getElementById('adm-fp-loc');
        if (fp) {
            fp.className = 'admin-fp' + (v ? ' active' : '');
            fp.textContent = v ? '📍 ' + v + ' ▾' : '📍 Location ▾';
        }
        document.getElementById('adm-dd-loc')?.classList.remove('open');
        admOpenDD = null;
        admApplyFilters();
    }

    function admSetRhs(v) {
        admFilterRhs = v;
        const fp = document.getElementById('adm-fp-rhs');
        if (fp) {
            fp.className = 'admin-fp' + (v ? ' active' : '');
            fp.textContent = v ? '🎨 ' + v + ' ▾' : '🎨 Grade ▾';
        }
        document.getElementById('adm-dd-rhs')?.classList.remove('open');
        admOpenDD = null;
        admApplyFilters();
    }

    function admClearDates() {
        document.getElementById('adm-date-from').value = '';
        document.getElementById('adm-date-to').value = '';
        admApplyFilters();
    }

    function admApplyFilters() {
        const search = admHistSearch.toLowerCase();
        const dateFrom = document.getElementById('adm-date-from')?.value;
        const dateTo = document.getElementById('adm-date-to')?.value;
        const today = new Date().toDateString();

        let filtered = admAllScans.filter(s => {
            if (admFilterStatus === 'verified' && !(s.verified === 1 || s.correction)) return false;
            if (admFilterStatus === 'pending' && (s.verified === 1 || s.correction)) return false;
            if (admFilterGrader) {
                const g = s.username || s.grader_name || '';
                if (g !== admFilterGrader) return false;
            }
            if (admFilterLoc && s.location !== admFilterLoc) return false;
            if (admFilterRhs && s.rhs_grade !== admFilterRhs) return false;
            if (dateFrom && s.scanned_at < dateFrom) return false;
            if (dateTo && s.scanned_at.slice(0, 10) > dateTo) return false;
            if (search) {
                const hay = [s.rhs_grade, s.username, s.grader_name, s.location, s.batch_id].join(' ').toLowerCase();
                if (!hay.includes(search)) return false;
            }
            return true;
        });

        const todayScans = filtered.filter(s => new Date(s.scanned_at).toDateString() === today);
        const verifiedScans = filtered.filter(s => s.verified === 1);
        const des = filtered.map(s => parseFloat(s.delta_e)).filter(n => !isNaN(n));
        const avgDe = des.length ? (des.reduce((a, b) => a + b, 0) / des.length).toFixed(1) : '—';

        document.getElementById('adm-hs-total').textContent = filtered.length;
        document.getElementById('adm-hs-today').textContent = todayScans.length;
        document.getElementById('adm-hs-verified').textContent = verifiedScans.length;
        document.getElementById('adm-hs-avgde').textContent = avgDe;

        renderAdmHistList(filtered);
    }

    function renderAdmHistList(items) {
        const el = document.getElementById('adm-hist-list');
        if (!items || items.length === 0) {
            el.innerHTML = '<div class="admin-empty">📭 No scans match filters.</div>';
            return;
        }

        // Group by date
        const groups = {};
        items.forEach(s => {
            const gk = new Date(s.scanned_at).toLocaleDateString('en-US', {
                weekday: 'short',
                month: 'short',
                day: 'numeric'
            });
            if (!groups[gk]) groups[gk] = [];
            groups[gk].push(s);
        });

        let html = '';
        for (const [date, scans] of Object.entries(groups)) {
            html += `<div class="admin-section-label">📅 ${date} · ${scans.length} scans</div>`;
            html += scans.map(s => {
                const deVal = s.delta_e != null ? parseFloat(s.delta_e) : null;
                const deColor = deVal == null ? '#8090a0' : deVal < 2 ? '#1a8c45' : deVal < 5 ? '#c47c00' : '#c0001a';
                const ts = new Date(s.scanned_at).toLocaleTimeString('en-US', {hour: '2-digit', minute: '2-digit'});
                const grader = s.username || s.grader_name || '—';
                const thumb = s.thumbnail_b64 && s.thumbnail_b64.startsWith('data:')
                        ? `<img src="${s.thumbnail_b64}" style="width:100%;height:100%;object-fit:cover;" alt="">`
                        : `<div style="position:absolute;inset:0;background:linear-gradient(135deg,${s.dominant_hex||'#c8ddd0'},${s.matched_hex||'#9abcaa'});"></div><span style="position:relative;z-index:1;font-size:.55rem;font-weight:800;color:#fff;font-family:IBM Plex Mono,monospace;">${s.rhs_grade||'?'}</span>`;
                const statusTag = s.verified === 1
                        ? '<span style="font-size:.58rem;padding:2px 6px;border-radius:4px;font-weight:700;background:#e8f5ee;color:#1a8c45;border:1px solid #c8e8d4;">✅ Verified</span>'
                        : s.correction
                                ? `<span style="font-size:.58rem;padding:2px 6px;border-radius:4px;font-weight:700;background:#e6f5f2;color:#0a8a76;border:1px solid #c0e8e0;">📝 ${s.correction}</span>`
                                : '<span style="font-size:.58rem;padding:2px 6px;border-radius:4px;font-weight:700;background:#f0f4fa;color:#6080a0;border:1px solid #d0d8e8;">⏳ Pending</span>';
                const storeKey = _storeAndKey(s);
                return `<div class="admin-hist-item" onclick="openDetailByKey('${storeKey}')">
        <div class="admin-hist-strip" style="background:${deColor}"></div>
        <div class="admin-hist-thumb">${thumb}</div>
        <div class="admin-hist-body">
          <div class="admin-hist-top">
            <span class="admin-hist-code">${s.rhs_grade||'?'}</span>
            <span class="admin-hist-de" style="color:${deColor};background:${deColor}12;border:1px solid ${deColor}30;">ΔE ${deVal!=null?deVal.toFixed(1):'—'}</span>
          </div>
          <div class="admin-hist-grader">👤 ${grader}</div>
          <div class="admin-hist-loc">📍 ${s.location||'No location'}</div>
          <div class="admin-hist-meta">${ts}${s.batch_id?' · '+s.batch_id:''} · ${statusTag}</div>
        </div>
        <div class="admin-hist-right">
          <div class="admin-hist-sw" style="background:${s.dominant_hex||'#c8ddd0'}"></div>
        </div>
      </div>`;
            }).join('');
        }
        el.innerHTML = html;
    }

    // ============================================================
    //  ADMIN USERS
    // ============================================================
    async function admLoadUsers() {
        try {
            const r = await fetch('/api/admin/users');
            admAllUsers = await r.json();
            admRenderUsers(admAllUsers);
        } catch (e) {
            document.getElementById('adm-users-list').innerHTML = '<div class="admin-empty">⚠️ Could not load users.</div>';
        }
    }

    function admSearchUsers(v) {
        admUserSearch = v.toLowerCase();
        const filtered = admAllUsers.filter(u => u.username && u.username.toLowerCase().includes(admUserSearch));
        admRenderUsers(filtered);
    }

    function admRenderUsers(users) {
        const el = document.getElementById('adm-users-list');
        if (!users || users.length === 0) {
            el.innerHTML = '<div class="admin-empty">👥 No users found.</div>';
            return;
        }

        el.innerHTML = users.map(u => {
            const initials = (u.username || '?').slice(0, 2).toUpperCase();
            const created = u.created_at ? new Date(u.created_at).toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                year: 'numeric'
            }) : '—';
            const lastActive = u.last_active ? new Date(u.last_active).toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric'
            }) : 'Never';
            const scans = u.scan_count || 0;
            const verified = u.verified_count || 0;
            const isOnline = u.is_online === true;
            const isRecent = false;
            const accuracy = scans > 0 ? Math.round(verified / scans * 100) : 0;

            const lastScanTs = admAllScans.filter(s => s.user_id === u.id).map(s => s.scanned_at).sort().pop();
            const lastScanAgo = (ts => {
                if (!ts) return null;
                const d = Date.now() - new Date(ts).getTime();
                const m = Math.floor(d / 60000);
                const h = Math.floor(d / 3600000);
                const dy = Math.floor(d / 86400000);
                return m < 1 ? 'just now' : m < 60 ? m + 'm ago' : h < 24 ? h + 'h ago' : dy + 'd ago';
            })(lastScanTs);
            return `<div class="admin-user-card">
      <div class="admin-user-header">
        <div class="admin-user-avatar">${initials}</div>
        <div class="admin-user-info">
          <div class="admin-user-name">${u.username}</div>
          <div class="admin-user-meta">Joined ${created} · Last active ${lastActive}</div>
        </div>
        <div class="admin-user-status" style="text-align:center;">
          <div class="admin-user-dot ${isOnline?'active':'inactive'}"></div>
          <span style="font-size:.65rem;color:${isOnline?'#1a8c45':'#c0001a'};font-weight:700;">${isOnline?'Online':'Offline'}</span>
          ${lastScanAgo?`<div style="font-size:.58rem;color:#888;margin-top:1px;">${lastScanAgo}</div>`
        :
            ''
        }
    </div>
    </div>
        <div class="admin-user-stats">
            <div class="admin-us">
                <div class="admin-us-num">${scans}</div>
                <div class="admin-us-lbl">Scans</div>
            </div>
            <div class="admin-us">
                <div class="admin-us-num">${verified}</div>
                <div class="admin-us-lbl">Verified</div>
            </div>
            <div class="admin-us">
                <div class="admin-us-num">${accuracy}%</div>
                <div class="admin-us-lbl">Accuracy</div>
            </div>
        </div>
        <div class="admin-user-actions">
            <button class="admin-ua-btn primary" onclick="admViewUserScans('${u.username}')">📊 View Scans</button>
            <button class="admin-ua-btn" onclick="admExportUser('${u.id}','${u.username}')">📥 Export</button>
            <button class="admin-ua-btn danger" onclick="admDeleteUser('${u.id}','${u.username}')">🗑 Delete</button>
        </div>
    </div>
        `;
  }).join('');
}

function _normalizeVerdict(v){
  if(!v)return'--';
  var l=v.toLowerCase().replace(/\s+/g,'');
  if(l==='strongmatch'||l==='strong')return'STRONG MATCH';
  if(l==='goodmatch'||l==='good')return'GOOD MATCH';
  if(l==='likelymatch'||l==='likely'||l==='fairmatch'||l==='fair')return'LIKELY MATCH';
  if(l==='weakmatch'||l==='weak'||l==='poormatch'||l==='poor'||l==='uncertain')return'UNCERTAIN';
  return v.toUpperCase();
}
function _verdictStyle(verdict){
  var l=verdict.toLowerCase();
  if(l.indexOf('strong')>=0)return{c:'#1a8c45',bg:'#e6f7ee'};
  if(l.indexOf('good')>=0)return{c:'#3a9e6e',bg:'#edf7f2'};
  if(l.indexOf('likely')>=0)return{c:'#b07d2e',bg:'#fff8e6'};
  return{c:'#8b6914',bg:'#fdf3dc'};
}
function admViewUserScans(username){
  const user=admAllUsers.find(function(u){return u.username===username;})||{};
  const userId=user.id||null;
  const scans=admAllScans.filter(function(s){return s.user_id===userId||s.username===username||s.grader_name===username;});
  document.getElementById('usm-avatar').textContent=(username||'?').slice(0,2).toUpperCase();
  document.getElementById('usm-username').textContent=username;
  var total=scans.length,verified=scans.filter(function(s){return s.verified===1;}).length;
  document.getElementById('usm-subtitle').textContent=total+' scan'+(total!==1?'s':'')+' · '+verified+' verified';
  const body=document.getElementById('usm-body');
  if(scans.length===0){
    body.innerHTML='<div class="usm-empty">No scans found for '+username+'</div>';
  } else {
    var html='';
    scans.forEach(function(s){
      var hex=s.dominant_hex||s.matched_hex||'#cccccc';
      var matchedHex=s.matched_hex||'#cccccc';
      var rhs=s.rhs_grade||'--';
      var de=s.delta_e!=null?parseFloat(s.delta_e).toFixed(2)+' dE':'--';
      var score=s.match_score!=null?parseFloat(s.match_score).toFixed(1)+'%':'--';
      var date=s.scanned_at?new Date(s.scanned_at).toLocaleString('en-US',{month:'short',day:'numeric',year:'numeric',hour:'2-digit',minute:'2-digit'}):'--';
      var verdict=_normalizeVerdict(s.verdict);
      var vs=_verdictStyle(verdict);
      // Extra details
      var detailsHtml='';
      if(s.location){detailsHtml+='<div class="usm-detail-row"><span class="usm-detail-icon">📍</span><span class="usm-detail-val">'+s.location+'</span></div>';}
      if(s.batch_id||s.supplier){
        var batchStr=(s.batch_id?'Batch: '+s.batch_id:'')+(s.batch_id&&s.supplier?' · ':'')+(s.supplier?s.supplier:'');
        detailsHtml+='<div class="usm-detail-row"><span class="usm-detail-icon">📦</span><span class="usm-detail-val">'+batchStr+'</span></div>';
      }
      if(s.grader_notes||s.notes){
        detailsHtml+='<div class="usm-detail-row"><span class="usm-detail-icon">📝</span><span class="usm-detail-val">'+(s.grader_notes||s.notes)+'</span></div>';
      }
      var correctionHtml='';
      if(s.correction&&s.correction.trim()&&s.correction.trim()!==rhs){
        correctionHtml='<div class="usm-correction"><span>'+rhs+'</span><span class="usm-correction-arrow">→</span><span style="color:#1a8c45;font-weight:800;">'+s.correction.trim().toUpperCase()+'</span></div>';
      }
      var verifiedHtml=s.verified===1?'<div class="usm-verified">✔ Verified'+(s.grader_name?' by '+s.grader_name:'')+'</div>':'';
      html+='<div class="usm-scan-card">'
        +'<div style="display:flex;align-items:flex-start;gap:10px;">'
          +'<div style="display:flex;flex-direction:column;gap:3px;flex-shrink:0;margin-top:2px;">'
            +'<div class="usm-swatch" style="background:'+hex+'" title="Scanned color"></div>'
            +'<div class="usm-swatch" style="background:'+matchedHex+';opacity:.7;" title="Matched RHS color"></div>'
          +'</div>'
          +'<div style="flex:1;min-width:0;">'
            +'<div style="display:flex;align-items:center;justify-content:space-between;gap:6px;flex-wrap:wrap;">'
              +'<div class="usm-rhs">'+rhs+'</div>'
              +'<span class="usm-verdict" style="background:'+vs.bg+';color:'+vs.c+'">'+verdict+'</span>'
            +'</div>'
            +'<div class="usm-meta" style="margin-top:3px;">'+date+'</div>'
            +'<div class="usm-de" style="margin-top:2px;">'+de+' / '+score+'</div>'
            +(verifiedHtml?'<div style="margin-top:3px;">'+verifiedHtml+'</div>':'')
            +(correctionHtml?'<div style="margin-top:5px;">'+correctionHtml+'</div>':'')
            +(detailsHtml?'<div class="usm-details">'+detailsHtml+'</div>':'')
          +'</div>'
        +'</div>'
        +'</div>';
    });
    body.innerHTML=html;
  }
  document.getElementById('user-scans-modal').classList.add('open');
}

function closeUserScansModal(){
  document.getElementById('user-scans-modal').classList.remove('open');
}

function admExportUser(userId,username){
  const userScans=admAllScans.filter(s=>s.user_id===userId||s.username===username||s.grader_name===username);
  if(userScans.length===0){alert('No scans found for '+username);return;}
  const headers=['Date','RHS Grade','Delta-E','Match Score','Verdict','Location','Batch ID','Verified','Correction'];
  const rows=userScans.map(s=>[
    new Date(s.scanned_at).toLocaleDateString(),
    s.rhs_grade||'',
    s.delta_e!=null?parseFloat(s.delta_e).toFixed(2):'',
    s.match_score!=null?parseFloat(s.match_score).toFixed(1)+'%':'',
    s.verdict||'',
    s.location||'',
    s.batch_id||'',
    s.verified===1?'Yes':'No',
    s.correction||''
  ]);
  const csv=[headers,...rows].map(r=>r.map(v=>'"'+String(v).replace(/"/g,'""')+'"').join(',')).join('\n');
  const blob=new Blob([csv],{type:'text/csv'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`
        ${username}_scans_${new Date().toISOString().slice(0,10)}.csv`;a.click();
  URL.revokeObjectURL(a.href);
}

async function admDeleteUser(userId,username){
  if(!confirm(`⚠️ DELETE
        user
        "${username}" ?\n\nThis
        will
        permanently
        delete the
        account
        and
        ALL
        their
        scans.\nThis
        cannot
        be
        undone.`))return;
  try{
    const r=await fetch('/api/admin/delete-user/'+userId,{method:'POST'});
    const d=await r.json();
    if(d.ok){
      admAllUsers=admAllUsers.filter(u=>u.id!==userId);
      admAllScans=admAllScans.filter(s=>s.user_id!==userId);
      admRenderUsers(admAllUsers.filter(u=>u.username&&u.username.toLowerCase().includes(admUserSearch)));
      // Show success toast
      showAdmToast('✅ User "'+username+'" deleted successfully.');
    }else{
      alert('Delete failed: '+(d.error||'Unknown error'));
    }
  }catch(e){alert('Error: '+e.message);}
}

function showAdmToast(msg){
  let t=document.getElementById('adm-toast');
  if(!t){t=document.createElement('div');t.id='adm-toast';t.style.cssText='display:none;position:fixed;top:80px;left:16px;right:16px;background:#1a2744;border:1px solid rgba(91,164,245,.3);border-radius:9px;padding:12px 16px;font-size:.82rem;font-weight:600;color:#fff;z-index:999;box-shadow:0 4px 20px rgba(0,0,0,.25);';document.body.appendChild(t);}
  t.textContent=msg;t.style.display='block';setTimeout(()=>{t.style.display='none';},3500);
}

// ============================================================
//  ADMIN SETTINGS
// ============================================================
async function admLoadSettings(){
  try{
    const [scans,users]=await Promise.all([
      fetch('/api/admin/scans').then(r=>r.json()).catch(()=>[]),
      fetch('/api/admin/users').then(r=>r.json()).catch(()=>[])
    ]);
    const verifiedCount=scans.filter(s=>s.verified===1).length;
    const needed=Math.max(0,1000-verifiedCount);
    const pct=Math.min(100,Math.round(verifiedCount/1000*100));
    const ready=verifiedCount>=1000;

    document.getElementById('adm-model-verified').textContent=verifiedCount;
    document.getElementById('adm-model-needed').textContent=needed;
    document.getElementById('adm-model-ready').textContent=ready?'✅ Ready':'⏳ Building';
    document.getElementById('adm-model-bar').style.width=pct+'%';
    document.getElementById('adm-retrain-btn').disabled=!ready;

    document.getElementById('adm-sys-users').textContent=users.length+' registered graders';
    document.getElementById('adm-sys-badge').textContent=users.length;

    // Check health
    const health=await fetch('/health').then(r=>r.json()).catch(()=>null);
    if(health){
      const dbBadge=document.getElementById('adm-db-badge');
      if(dbBadge){dbBadge.textContent=health.supabase?'✅ Supabase':'SQLite only';dbBadge.className='admin-sr-badge'+(health.supabase?' ok':'');}
    }
  }catch(e){console.error('Admin settings load failed',e);}
}

async function admTriggerRetrain(){
  const btn=document.getElementById('adm-retrain-btn');
  btn.disabled=true;btn.textContent='⏳ Training...';
  const existing=document.getElementById('retrain-report');if(existing)existing.remove();
  let retrainData=null;
  try{
    const r=await fetch('/api/admin/retrain',{method:'POST'});
    const d=await r.json();
    retrainData=d;
    btn.disabled=false;
    if(!d.ok){
      btn.textContent='❌ Failed';
      showAdmToast('⚠️ '+d.message);
      setTimeout(()=>btn.textContent='🔁 Trigger Model Retraining',3000);
      return;
    }
    btn.textContent='✅ Retrain Complete';
    setTimeout(()=>btn.textContent='🔁 Trigger Model Retraining',4000);
    showAdmToast('✅ Models retrained and live! No restart needed.');

    // ── Build retrain report ─────────────────────────────────────────
    const maxC = d.top_misclassifications && d.top_misclassifications.length ? d.top_misclassifications[0].count : 1;
    const avgAcc = d.mlp_a_accuracy && d.mlp_b_accuracy && d.mlp_c_accuracy && d.mlp_d_accuracy
      ? ((d.mlp_a_accuracy + d.mlp_b_accuracy + d.mlp_c_accuracy + d.mlp_d_accuracy) / 4).toFixed(2)
      : '—';

    const errorsHtml = d.top_misclassifications && d.top_misclassifications.length
      ? `<div style="font-size:.6rem;color:rgba(255,255,255,.4);margin-bottom:8px;text-transform:uppercase;letter-spacing:.06em;font-weight:700;">Top Grade Corrections Applied</div>`
        + d.top_misclassifications.map(m => `
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
            <div style="font-family:IBM Plex Mono,monospace;font-size:.76rem;font-weight:700;color:#fff;min-width:90px;">${m.pattern}</div>
            <div style="flex:1;height:5px;background:rgba(255,255,255,.1);border-radius:99px;overflow:hidden;">
              <div style="height:100%;width:${Math.round((m.count/maxC)*100)}%;background:linear-gradient(90deg,#3b7dd8,#5ba4f5);border-radius:99px;"></div>
            </div>
            <div style="font-size:.65rem;color:rgba(255,255,255,.5);min-width:22px;text-align:right;">${m.count}×</div>
          </div>`).join('')
      : `<div style="font-size:.72rem;color:rgba(255,255,255,.35);text-align:center;padding:8px;">No corrections recorded yet — trained on swatch data only.</div>`;

    const reportHtml = `
    <div id="retrain-report" style="margin-top:16px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:12px;padding:16px;">

      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
        <div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.1em;color:rgba(255,255,255,.5);font-weight:700;">📊 Retrain Report</div>
        <div style="font-size:.65rem;color:rgba(255,255,255,.35);font-family:IBM Plex Mono,monospace;">${new Date().toLocaleString()} · ${d.elapsed_seconds}s</div>
      </div>

      <!-- Summary stats -->
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:10px;">
        <div style="background:rgba(255,255,255,.07);border-radius:9px;padding:10px;text-align:center;">
          <div style="font-size:1.4rem;font-weight:800;color:#fff;font-family:IBM Plex Mono,monospace;line-height:1;">${d.verified_count}</div>
          <div style="font-size:.5rem;color:rgba(255,255,255,.4);margin-top:3px;text-transform:uppercase;letter-spacing:.05em;">Verified Scans</div>
        </div>
        <div style="background:rgba(255,255,255,.07);border-radius:9px;padding:10px;text-align:center;">
          <div style="font-size:1.4rem;font-weight:800;color:#fff;font-family:IBM Plex Mono,monospace;line-height:1;">${(d.total_training_samples||0).toLocaleString()}</div>
          <div style="font-size:.5rem;color:rgba(255,255,255,.4);margin-top:3px;text-transform:uppercase;letter-spacing:.05em;">Total Samples</div>
        </div>
        <div style="background:rgba(255,255,255,.07);border-radius:9px;padding:10px;text-align:center;">
          <div style="font-size:1.4rem;font-weight:800;color:#fff;font-family:IBM Plex Mono,monospace;line-height:1;">${d.classes}</div>
          <div style="font-size:.5rem;color:rgba(255,255,255,.4);margin-top:3px;text-transform:uppercase;letter-spacing:.05em;">RHS Classes</div>
        </div>
      </div>

      <!-- Data source breakdown -->
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:10px;">
        <div style="background:rgba(74,222,128,.12);border:1px solid rgba(74,222,128,.2);border-radius:8px;padding:8px;text-align:center;">
          <div style="font-size:1.1rem;font-weight:800;color:#4ade80;font-family:IBM Plex Mono,monospace;">${d.real_photos_used||0}</div>
          <div style="font-size:.48rem;color:rgba(255,255,255,.4);margin-top:2px;text-transform:uppercase;">📷 Real Photos</div>
        </div>
        <div style="background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.2);border-radius:8px;padding:8px;text-align:center;">
          <div style="font-size:1.1rem;font-weight:800;color:#fbbf24;font-family:IBM Plex Mono,monospace;">${d.synthetic_fallbacks||0}</div>
          <div style="font-size:.48rem;color:rgba(255,255,255,.4);margin-top:2px;text-transform:uppercase;">🧪 Synthetic</div>
        </div>
        <div style="background:rgba(255,255,255,.05);border-radius:8px;padding:8px;text-align:center;">
          <div style="font-size:1.1rem;font-weight:800;color:rgba(255,255,255,.5);font-family:IBM Plex Mono,monospace;">${d.skipped_errors||0}</div>
          <div style="font-size:.48rem;color:rgba(255,255,255,.35);margin-top:2px;text-transform:uppercase;">⚠️ Skipped</div>
        </div>
      </div>

      <!-- MLP accuracy — all 4 models + ensemble avg -->
      <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:9px;padding:10px;margin-bottom:10px;">
        <div style="font-size:.55rem;text-transform:uppercase;letter-spacing:.08em;color:rgba(255,255,255,.4);margin-bottom:8px;font-weight:700;">Model Accuracy (Train Set)</div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:8px;">
          ${['a','b','c','d'].map(m=>`
            <div style="text-align:center;">
              <div style="font-size:1rem;font-weight:800;color:#4ade80;font-family:IBM Plex Mono,monospace;">${d['mlp_'+m+'_accuracy']||'—'}%</div>
              <div style="font-size:.5rem;color:rgba(255,255,255,.35);margin-top:2px;text-transform:uppercase;">MLP-${m.toUpperCase()}</div>
            </div>`).join('')}
        </div>
        <div style="border-top:1px solid rgba(255,255,255,.08);padding-top:8px;display:flex;align-items:center;justify-content:space-between;">
          <div style="font-size:.58rem;color:rgba(255,255,255,.4);text-transform:uppercase;letter-spacing:.06em;">Ensemble Average</div>
          <div style="font-size:1.1rem;font-weight:800;color:#5ba4f5;font-family:IBM Plex Mono,monospace;">${avgAcc}%</div>
        </div>
      </div>

      <!-- Top corrections -->
      <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:9px;padding:10px;margin-bottom:12px;">
        ${errorsHtml}
      </div>

      <div style="font-size:.62rem;color:rgba(255,255,255,.3);line-height:1.6;margin-bottom:12px;">${d.note||''}</div>

      <!-- Action buttons -->
      <div style="display:flex;gap:8px;">
        <button id="retrain-dl-csv"
          style="flex:1;padding:10px;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);border-radius:8px;color:#fff;font-size:.78rem;font-weight:700;cursor:pointer;font-family:IBM Plex Mono,monospace;">
          ⬇️ CSV Report
        </button>
        <button id="retrain-dl-json"
          style="flex:1;padding:10px;background:rgba(59,125,216,.25);border:1px solid rgba(59,125,216,.4);border-radius:8px;color:#5ba4f5;font-size:.78rem;font-weight:700;cursor:pointer;font-family:IBM Plex Mono,monospace;">
          ⬇️ JSON Data
        </button>
      </div>
    </div>`;

    const card = btn.closest('.model-train-card') || btn.parentElement;
    card.insertAdjacentHTML('beforeend', reportHtml);

    // CSV download
    document.getElementById('retrain-dl-csv').onclick = async function() {
      this.textContent = '⏳ Generating...'; this.disabled = true;
      try {
        const resp = await fetch('/api/admin/retrain/export', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(retrainData)
        });
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `abaca_retrain_${new Date().toISOString().slice(0,10)}.csv`;
        a.click(); URL.revokeObjectURL(url);
        this.textContent = '✅ Downloaded';
        setTimeout(() => { this.textContent = '⬇️ CSV Report'; this.disabled = false; }, 2000);
      } catch(e) { this.textContent = '❌ Failed'; this.disabled = false; }
    };

    // JSON download
    document.getElementById('retrain-dl-json').onclick = function() {
      const blob = new Blob([JSON.stringify(retrainData, null, 2)], {type: 'application/json'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `abaca_retrain_${new Date().toISOString().slice(0,10)}.json`;
      a.click(); URL.revokeObjectURL(url);
      this.textContent = '✅ Downloaded';
      setTimeout(() => this.textContent = '⬇️ JSON Data', 2000);
    };

  }catch(e){
    btn.disabled=false;btn.textContent='🔁 Trigger Model Retraining';
    showAdmToast('⚠️ Error: '+e.message);
  }
}


// ============================================================