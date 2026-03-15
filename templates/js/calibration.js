//  CALIBRATION
    // ============================================================
    function loadWbProfile() {
        try {
            const saved = localStorage.getItem('abaca_wb_profile');
            if (saved) wbProfile = JSON.parse(saved);
        } catch (e) {
            wbProfile = null;
        }
        updateWbUI();
    }

    function updateWbUI() {
        const badge = document.getElementById('wb-status-badge'), lastEl = document.getElementById('wb-last');
        if (!badge || !lastEl) return;
        if (wbProfile) {
            badge.textContent = '✅ Calibrated';
            badge.className = 'sr-badge ok';
            const d = new Date(wbProfile.date);
            lastEl.textContent = 'Last: ' + d.toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                year: 'numeric'
            });
        } else {
            badge.textContent = 'Auto';
            badge.className = 'sr-badge';
            lastEl.textContent = 'Last: Never calibrated';
        }
    }

    function openCalibration() {
        goCalStep1();
        loadCalProfileDisplay();
        document.getElementById('cal-modal').style.display = 'block';
        document.getElementById('cal-modal').scrollTop = 0;
    }

    function closeCalibration() {
        document.getElementById('cal-modal').style.display = 'none';
        loadSettings();
    }

    function setCalStep(n) {
        [1, 2, 3].forEach(i => {
            document.getElementById('cal-step-' + i).style.display = i === n ? 'block' : 'none';
            const dot = document.getElementById('cdot-' + i);
            if (dot) dot.className = 'cal-step-dot' + (i <= n ? ' active' : '');
        });
        document.getElementById('cal-step-label').textContent = 'Step ' + n + ' of 3';
    }

    function goCalStep1() {
        setCalStep(1);
        loadCalProfileDisplay();
    }

    function loadCalProfileDisplay() {
        const el = document.getElementById('cal-profile-display');
        if (!el) return;
        if (wbProfile) {
            const d = new Date(wbProfile.date);
            el.innerHTML = 'R:' + wbProfile.r.toFixed(3) + ' G:' + wbProfile.g.toFixed(3) + ' B:' + wbProfile.b.toFixed(3) + ' · Saved ' + d.toLocaleDateString();
        } else {
            el.innerHTML = '<div style="color:var(--sub);font-size:.78rem;">No calibration profile saved yet.</div>';
        }
    }

    function startCalStep2() {
        document.getElementById('cal-cam-input').click();
    }

    function handleCalImage(inp) {
        const f = inp.files[0];
        if (!f) return;
        inp.value = '';
        const img = new Image();
        img.onload = function () {
            const canvas = document.getElementById('cal-preview-canvas');
            const MAX = 600, scale = Math.min(MAX / img.naturalWidth, MAX / img.naturalHeight, 1);
            canvas.width = Math.round(img.naturalWidth * scale);
            canvas.height = Math.round(img.naturalHeight * scale);
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            const x0 = Math.floor(canvas.width * .30), y0 = Math.floor(canvas.height * .30),
                    x1 = Math.floor(canvas.width * .70), y1 = Math.floor(canvas.height * .70);
            const pixels = ctx.getImageData(x0, y0, x1 - x0, y1 - y0).data;
            let rSum = 0, gSum = 0, bSum = 0, n = 0;
            const allR = [], allG = [], allB = [];
            for (let i = 0; i < pixels.length; i += 4) {
                allR.push(pixels[i]);
                allG.push(pixels[i + 1]);
                allB.push(pixels[i + 2]);
            }
            allR.sort((a, b) => a - b);
            allG.sort((a, b) => a - b);
            allB.sort((a, b) => a - b);
            const trim = Math.floor(allR.length * .05);
            for (let i = trim; i < allR.length - trim; i++) {
                rSum += allR[i];
                gSum += allG[i];
                bSum += allB[i];
                n++;
            }
            const mr = Math.round(rSum / n), mg = Math.round(gSum / n), mb = Math.round(bSum / n);
            const TARGET = 245;
            const gr = Math.min(TARGET / mr, 2.0), gg = Math.min(TARGET / mg, 2.0), gb = Math.min(TARGET / mb, 2.0);
            let bias = 'Neutral';
            if (mr < mg - 10 && mb > mg + 10) bias = '🔵 Cool/Blue cast'; else if (mr > mg + 10 && mb < mg - 10) bias = '🟠 Warm/Yellow cast';
            const avgBright = (mr + mg + mb) / 3;
            const maxGain = Math.max(gr, gg, gb);
            const warnings = [];
            if (avgBright < 150) warnings.push('⚠️ Too dark — retake on a brighter white surface');
            if (avgBright > 252) warnings.push('ℹ️ Already near-white — calibration may not be needed');
            if (maxGain > 1.8) warnings.push('⚠️ Strong correction — lighting is very uneven');
            calImageData = {mr, mg, mb, gr, gg, gb, bias, warnings, avgBright};
            document.getElementById('cal-measured-r').textContent = mr;
            document.getElementById('cal-measured-g').textContent = mg;
            document.getElementById('cal-measured-b').textContent = mb;
            const hex = '#' + mr.toString(16).padStart(2, '0') + mg.toString(16).padStart(2, '0') + mb.toString(16).padStart(2, '0');
            document.getElementById('cal-measured-swatch').style.background = hex;
            document.getElementById('cal-measured-hex').textContent = hex.toUpperCase();
            document.getElementById('cal-gain-r').textContent = gr.toFixed(3) + '×';
            document.getElementById('cal-gain-g').textContent = gg.toFixed(3) + '×';
            document.getElementById('cal-gain-b').textContent = gb.toFixed(3) + '×';
            document.getElementById('cal-bias-label').textContent = 'Detected: ' + bias;
            document.getElementById('cal-warnings').innerHTML = calImageData.warnings.map(w => `<div style="color:var(--warn);font-size:.72rem;margin-top:5px;padding:5px 8px;background:var(--warn-bg);border-radius:6px;">${w}</div>`).join('');
            setCalStep(2);
            URL.revokeObjectURL(img.src);
        };
        img.src = URL.createObjectURL(f);
    }

    function applyCalibration() {
        if (!calImageData) return;
        if (calImageData.avgBright < 150) {
            document.getElementById('cal-warnings').innerHTML = '<div style="color:var(--danger);font-size:.72rem;padding:6px 8px;background:var(--danger-bg);border-radius:6px;">❌ Cannot apply — reference too dark. Tap Retake and use a brighter white surface.</div>';
            return;
        }
        wbProfile = {
            r: calImageData.gr,
            g: calImageData.gg,
            b: calImageData.gb,
            bias: calImageData.bias,
            date: new Date().toISOString()
        };
        localStorage.setItem('abaca_wb_profile', JSON.stringify(wbProfile));
        document.getElementById('cal-saved-summary').innerHTML = '<div style="text-align:center;background:var(--card2);border-radius:7px;padding:8px;"><div style="font-family:IBM Plex Mono,monospace;font-weight:700;color:#e05050;">' + wbProfile.r.toFixed(3) + '×</div><div style="font-size:.6rem;color:var(--sub);">R gain</div></div><div style="text-align:center;background:var(--card2);border-radius:7px;padding:8px;"><div style="font-family:IBM Plex Mono,monospace;font-weight:700;color:#30a050;">' + wbProfile.g.toFixed(3) + '×</div><div style="font-size:.6rem;color:var(--sub);">G gain</div></div><div style="text-align:center;background:var(--card2);border-radius:7px;padding:8px;"><div style="font-family:IBM Plex Mono,monospace;font-weight:700;color:#3060c0;">' + wbProfile.b.toFixed(3) + '×</div><div style="font-size:.6rem;color:var(--sub);">B gain</div></div>';
        document.getElementById('cal-saved-time').textContent = 'Saved ' + new Date().toLocaleString() + ' · ' + wbProfile.bias;
        setCalStep(3);
        updateWbUI();
    }

    function applyWbToCanvas(canvas) {
        if (!wbProfile) return;
        const ctx = canvas.getContext('2d');
        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imgData.data;
        const {r: gr, g: gg, b: gb} = wbProfile;
        for (let i = 0; i < data.length; i += 4) {
            data[i] = Math.min(255, Math.round(data[i] * gr));
            data[i + 1] = Math.min(255, Math.round(data[i + 1] * gg));
            data[i + 2] = Math.min(255, Math.round(data[i + 2] * gb));
        }
        ctx.putImageData(imgData, 0, 0);
    }

    // ============================================================