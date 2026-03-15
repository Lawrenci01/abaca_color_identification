//  STATE
    // ============================================================
    let currentUser = null, sessionLocation = '', sessionScanCount = 0;

    function startHeartbeat() {
        if (!currentUser || !currentUser.id) return;
        const hb = () => fetch('/api/heartbeat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({user_id: currentUser.id})
        });
        hb();
        setInterval(hb, 60000);
    }

    let loginPin = '', regPin = '', adminPin = '';
    let box = {x: 0, y: 0, w: 0, h: 0}, dragging = null, dragStart = null;
    let croppedBlob = null, liveTimer = null, qualIssues = [];
    let camStream = null, currentResult = null, currentScanId = null;
    let verifyState = null, scanHistory = [];
    let allLoadedScans = [], openDropdown = null;
    let activeFilters = {status: 'all', rhs: '', location: '', grader: ''};
    let wbProfile = null, calImageData = null;
    let admAllScans = [], admAllUsers = [];
    let admFilterStatus = 'all', admFilterGrader = '', admFilterLoc = '', admFilterRhs = '';
    let admHistSearch = '', admUserSearch = '';
    const _scanStore = {};

    // ============================================================
    //  LOCATION / SESSION
    // ============================================================
    function getFullLocation() {
        return sessionLocation ? 'TCL — ' + sessionLocation : '';
    }

    function onLocInput() {
        sessionLocation = document.getElementById('sb-loc-input').value.trim();
        updateSessionBar();
        try {
            sessionStorage.setItem('abaca_loc', sessionLocation);
        } catch (e) {
        }
    }

    function onLocBlur() {
        sessionLocation = document.getElementById('sb-loc-input').value.trim();
        updateSessionBar();
    }

    function updateSessionBar() {
        if (!currentUser) return;
        document.getElementById('session-bar').style.display = 'flex';
        const name = currentUser.username;
        const initials = name.split(/[_\s]/).map(p => p[0] || '').join('').toUpperCase().slice(0, 2) || name[0].toUpperCase();
        document.getElementById('sb-avatar').textContent = initials;
        document.getElementById('sb-grader-name').textContent = name;
        document.getElementById('sb-scan-count').textContent = sessionScanCount;
        const slv = document.getElementById('settings-location-val');
        if (slv) slv.textContent = getFullLocation() || 'Not set';
    }

    function focusLocationBar() {
        switchScreen('scanner');
        setTimeout(() => {
            const inp = document.getElementById('sb-loc-input');
            if (inp) {
                inp.focus();
                inp.select();
            }
        }, 200);
    }

    function showLocToast() {
        const t = document.getElementById('loc-toast');
        t.style.display = 'block';
        setTimeout(() => {
            t.style.display = 'none';
        }, 3500);
    }

    // ============================================================
    //  SCREEN NAVIGATION
    // ============================================================
    function switchScreen(name) {
        document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        const sc = document.getElementById('screen-' + name);
        if (sc) sc.classList.add('active');
        const ni = document.getElementById('nav-' + name);
        if (ni) ni.classList.add('active');
        if (name === 'dashboard') loadDashboard();
        if (name === 'history') loadHistory();
        if (name === 'settings') loadSettings();
    }

    function goToScanner() {
        switchScreen('scanner');
    }

    function goToHistory() {
        switchScreen('history');
    }

    function setGreeting() {
        const h = new Date().getHours();
        const g = h < 12 ? 'Good morning' : 'Good afternoon';
        const suffix = h >= 17 ? '🌙' : h >= 12 ? '🌞' : '🌿';
        document.getElementById('dash-greeting').innerHTML = `${g} <span>${suffix}</span>`;
        document.getElementById('dash-date').textContent = new Date().toLocaleDateString('en-US', {
            weekday: 'long',
            month: 'long',
            day: 'numeric'
        });
    }

    // ============================================================