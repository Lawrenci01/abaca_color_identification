//  HISTORY + FILTERS
    // ============================================================
    let histSearchTimeout = null;

    function searchHistory(q) {
        clearTimeout(histSearchTimeout);
        histSearchTimeout = setTimeout(() => applyFilters(), 300);
    }

    async function loadHistory() {
        const el = document.getElementById('history-list-items');
        el.innerHTML = '<div class="hist-loading">Loading...</div>';
        try {
            const uid = currentUser ? currentUser.id : '';
            const r = await fetch('/api/scans?limit=200&user_id=' + uid);
            if (!r.ok) throw new Error('Failed');
            const d = await r.json();
            const stats = d.stats || {};
            const unsavedSession = scanHistory.filter(e => e._unsaved);
            allLoadedScans = [...unsavedSession, ...(d.scans || [])];
            allLoadedScans.forEach(s => _storeAndKey(s));
            document.getElementById('hs-total').textContent = (stats.total ?? 0) + unsavedSession.length;
            document.getElementById('hs-today').textContent = (stats.today ?? 0) + unsavedSession.length;
            document.getElementById('hs-verified').textContent = stats.verified ?? '—';
            document.getElementById('hs-avgde').textContent = stats.avg_de ?? '—';
            buildFilterDropdowns(allLoadedScans);
            applyFilters();
        } catch (e) {
            el.innerHTML = '<div class="hist-empty-v2">⚠️ Could not load history</div>';
        }
    }

    function _storeAndKey(s) {
        const k = s._session_id || s.id || ('k_' + Math.random().toString(36).slice(2, 10));
        s._storeKey = k;
        _scanStore[k] = s;
        return k;
    }

    function openDetailByKey(k) {
        const s = _scanStore[k];
        if (!s) return;
        openDetail(s._session_id || s.id || k, s);
    }

    function toggleDropdown(which) {
        if (openDropdown && openDropdown !== which) document.getElementById('fdd-' + openDropdown)?.classList.remove('open');
        const dd = document.getElementById('fdd-' + which);
        if (!dd) return;
        if (openDropdown === which) {
            dd.classList.remove('open');
            openDropdown = null;
        } else {
            dd.classList.add('open');
            openDropdown = which;
        }
    }

    document.addEventListener('click', e => {
        if (!e.target.closest('.filter-dropdown-wrap')) {
            document.querySelectorAll('.filter-dropdown').forEach(d => d.classList.remove('open'));
            openDropdown = null;
        }
    });

    function setStatusFilter(v) {
        activeFilters.status = v;
        ['all', 'verified', 'pending'].forEach(k => {
            document.getElementById('fp-' + k).classList.toggle('active', k === v);
        });
        applyFilters();
    }

    function setRhsFilter(v) {
        activeFilters.rhs = v;
        const fp = document.getElementById('fp-rhs');
        fp.className = 'filter-pill' + (v ? ' active' : '');
        fp.textContent = v ? '🎨 ' + v + ' ▾' : '🎨 RHS ▾';
        document.getElementById('fdd-rhs').classList.remove('open');
        openDropdown = null;
        applyFilters();
    }

    function setLocFilter(v) {
        activeFilters.location = v;
        const fp = document.getElementById('fp-loc');
        fp.className = 'filter-pill' + (v ? ' active' : '');
        fp.textContent = v ? '📍 ' + v + ' ▾' : '📍 Location ▾';
        document.getElementById('fdd-loc').classList.remove('open');
        openDropdown = null;
        applyFilters();
    }

    function setGraderFilter(v) {
        activeFilters.grader = v;
        const fp = document.getElementById('fp-grader');
        fp.className = 'filter-pill' + (v ? ' active' : '');
        fp.textContent = v ? '👤 ' + v + ' ▾' : '👤 Grader ▾';
        document.getElementById('fdd-grader').classList.remove('open');
        openDropdown = null;
        applyFilters();
    }

    function clearAllFilters() {
        activeFilters = {status: 'all', rhs: '', location: '', grader: ''};
        setStatusFilter('all');
        setRhsFilter('');
        setLocFilter('');
        setGraderFilter('');
        document.getElementById('history-search-input').value = '';
        applyFilters();
    }

    function buildFilterDropdowns(scans) {
        const rhsCodes = [...new Set(scans.map(s => s.rhs_grade).filter(Boolean))].sort();
        const locations = [...new Set(scans.map(s => s.location).filter(Boolean))].sort();
        const graders = [...new Set(scans.map(s => s.grader_name).filter(Boolean))].sort();
        const dRhs = document.getElementById('fdd-rhs');
        dRhs.innerHTML = '<div class="fdd-item" onclick="setRhsFilter(\'\')">All Grades</div>' + rhsCodes.map(c => `<div class="fdd-item" data-val="${c}" onclick="setRhsFilter('${c}')">${c}</div>`).join('');
        const dLoc = document.getElementById('fdd-loc');
        dLoc.innerHTML = '<div class="fdd-item" onclick="setLocFilter(\'\')">All Locations</div>' + locations.map(l => `<div class="fdd-item" data-val="${l}" onclick="setLocFilter('${l}')">${l}</div>`).join('');
        const dGrader = document.getElementById('fdd-grader');
        dGrader.innerHTML = '<div class="fdd-item" onclick="setGraderFilter(\'\')">All Graders</div>' + graders.map(g => `<div class="fdd-item" data-val="${g}" onclick="setGraderFilter('${g}')">${g}</div>`).join('');
    }

    function applyFilters() {
        const search = document.getElementById('history-search-input').value.toLowerCase();
        let filtered = allLoadedScans.filter(s => {
            if (activeFilters.status === 'verified' && s.verified !== 1) return false;
            if (activeFilters.status === 'pending' && s.verified === 1) return false;
            if (activeFilters.rhs && s.rhs_grade !== activeFilters.rhs) return false;
            if (activeFilters.location && s.location !== activeFilters.location) return false;
            if (activeFilters.grader && s.grader_name !== activeFilters.grader) return false;
            if (search) {
                const hay = [s.rhs_grade, s.batch_id, s.location, s.grader_name].join(' ').toLowerCase();
                if (!hay.includes(search)) return false;
            }
            return true;
        });
        document.getElementById('filter-count-chip').textContent = filtered.length + ' results';
        const hasFilter = activeFilters.status !== 'all' || activeFilters.rhs || activeFilters.location || activeFilters.grader || search;
        const banner = document.getElementById('active-filter-banner');
        banner.classList.toggle('show', hasFilter);
        if (hasFilter) {
            const parts = [];
            if (activeFilters.status !== 'all') parts.push(activeFilters.status === 'verified' ? '✅ Verified' : '⏳ Pending');
            if (activeFilters.rhs) parts.push('RHS: ' + activeFilters.rhs);
            if (activeFilters.location) parts.push('📍 ' + activeFilters.location);
            if (activeFilters.grader) parts.push('👤 ' + activeFilters.grader);
            if (search) parts.push('"' + search + '"');
            document.getElementById('active-filter-text').textContent = 'Showing: ' + parts.join(' · ');
        }
        renderHistoryList(filtered);
    }

    function renderHistoryList(items) {
        const el = document.getElementById('history-list-items');
        if (!items || items.length === 0) {
            el.innerHTML = '<div class="hist-empty-v2">📭 No scans match current filters.</div>';
            return;
        }
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
            html += `<div class="hist-section-label">📅 ${date}</div>`;
            html += scans.map(s => {
                const thumb = s.thumbnail_b64 && s.thumbnail_b64.startsWith('data:')
                        ? `<img class="hist-thumb-v2" src="${s.thumbnail_b64}" alt="">`
                        : `<div class="hist-no-thumb"><div style="position:absolute;inset:0;background:linear-gradient(145deg,${s.dominant_hex||'#c8ddd0'},${s.matched_hex||'#9abcaa'});"></div><span style="position:relative;z-index:1;font-size:.62rem;font-weight:800;color:#fff;font-family:IBM Plex Mono,monospace;text-shadow:0 1px 3px rgba(0,0,0,.5);">${s.rhs_grade||'?'}</span></div>`;
                const tag = s._unsaved ? '<span class="hist-tag unsaved">📱 Unsaved</span>' : s.verified === 1 ? '<span class="hist-tag verified">✅ Verified</span>' : s.correction ? `<span class="hist-tag edited">📝 ${s.correction}</span>` : '<span class="hist-tag pending">⏳ Pending</span>';
                const ts = new Date(s.scanned_at).toLocaleTimeString('en-US', {hour: '2-digit', minute: '2-digit'});
                const deColor = s.verdict_color || '#1a8c45';
                return `<div class="hist-item-v2" onclick="openDetailByKey('${s._storeKey}')">
        <div class="hist-color-strip" style="background:${s.verdict_color||'var(--accent)'};"></div>
        ${thumb}
        <div class="hist-body-v2">
          <div class="hist-top-row"><span class="hist-code-v2">${s.rhs_grade||'?'}</span><span class="hist-de-badge" style="color:${deColor};border-color:${deColor}22;background:${deColor}10;">ΔE ${s.delta_e??'—'}</span></div>
          <div class="hist-grader-loc"><div class="hist-grader-pill">👤 ${s.grader_name||'—'}</div><span style="color:var(--border);font-size:.6rem;">·</span><div class="hist-loc-pill">📍 ${s.location||'—'}</div></div>
          <div class="hist-meta">${ts}${s.batch_id?' · '+s.batch_id:''}</div>
          <div class="hist-tags">${tag}</div>
        </div>
        <div class="hist-right"><div class="hist-sw-v2" style="background:${s.dominant_hex||'#d6e5da'}"></div></div>
      </div>`;
            }).join('');
        }
        el.innerHTML = html;
    }

    // ============================================================