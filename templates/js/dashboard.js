//  DASHBOARD
    // ============================================================
    async function loadDashboard() {
        setGreeting();
        try {
            const uid = currentUser ? currentUser.id : '';
            const r = await fetch('/api/scans?user_id=' + uid);
            if (!r.ok) return;
            const d = await r.json();
            const stats = d.stats || {};
            document.getElementById('stat-today').textContent = stats.today ?? '0';
            document.getElementById('stat-total').textContent = stats.total ?? '0';
            document.getElementById('stat-avgde').textContent = stats.avg_de ?? '—';
            document.getElementById('topbar-scan-count').textContent = (stats.total ?? 0) + ' scans';
            document.getElementById('stat-verified-pct').textContent = (stats.total > 0 ? Math.round((stats.verified / stats.total) * 100) : 0) + '% verified';
            const grades = (d.scans || []).map(s => s.rhs_grade).filter(Boolean);
            const topGrade = grades.length ? Object.entries(grades.reduce((a, g) => {
                a[g] = (a[g] || 0) + 1;
                return a;
            }, {})).sort((a, b) => b[1] - a[1])[0][0] : '—';
            document.getElementById('stat-topgrade').textContent = topGrade;
            const recent = d.recent || (d.scans || []).slice(0, 5);
            renderRecentList(recent);
            document.getElementById('sync-dot').className = 'sync-dot online';
        } catch (e) {
            console.error('Dashboard load failed', e);
        }
    }

    function loadSettings() {
        if (currentUser) document.getElementById('settings-username').textContent = '👤 ' + currentUser.username;
        const slv = document.getElementById('settings-location-val');
        if (slv) slv.textContent = getFullLocation() || 'Not set';
        loadWbProfile();
    }

    function renderRecentList(items) {
        const el = document.getElementById('recent-list');
        if (!items || items.length === 0) {
            el.innerHTML = '<div class="dash-empty">📭 No scans yet.<br>Tap START SCAN to begin.</div>';
            return;
        }
        items.forEach(s => _storeAndKey(s));
        el.innerHTML = items.map(s => {
            const thumb = s.thumbnail_b64 && s.thumbnail_b64.startsWith('data:')
                    ? `<img class="ri-thumb" src="${s.thumbnail_b64}" alt="">`
                    : `<div class="ri-no-thumb"><div style="position:absolute;inset:0;background:linear-gradient(145deg,${s.dominant_hex||'#c8ddd0'},${s.matched_hex||'#9abcaa'});border-radius:9px;"></div><span style="position:relative;z-index:1;font-size:.52rem;font-weight:800;color:#fff;font-family:IBM Plex Mono,monospace;text-shadow:0 1px 2px rgba(0,0,0,.5);">${s.rhs_grade||'?'}</span></div>`;
            const badge = s.verified === 1 ? '<span class="ri-badge ok">✅</span>' : '';
            const ts = new Date(s.scanned_at).toLocaleTimeString('en-US', {hour: '2-digit', minute: '2-digit'});
            const locStr = s.location ? ' · ' + s.location : '';
            return `<div class="recent-item" onclick="openDetailByKey('${s._storeKey}')">
      ${thumb}
      <div class="ri-body"><div class="ri-code">${s.rhs_grade||'?'}${badge}</div><div class="ri-sub">${ts}${locStr} · ΔE ${s.delta_e??'—'}</div></div>
      <div class="ri-sw" style="background:${s.dominant_hex||'#ccc'}"></div>
    </div>`;
        }).join('');
    }

    // ============================================================