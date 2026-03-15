//  CAMERA + IMAGE HANDLING
    // ============================================================
    function openCamera() {
        if (/Mobi|Android|iPhone|iPad/i.test(navigator.userAgent)) {
            document.getElementById('file-cam').click();
            return;
        }
        if (!navigator.mediaDevices?.getUserMedia) {
            document.getElementById('file-cam').click();
            return;
        }
        const camConstraints = {
            video: {
                facingMode: 'environment',
                width: {ideal: 4096},
                height: {ideal: 3072},
                advanced: [{zoom: 1}, {focusMode: 'continuous'}, {exposureMode: 'continuous'}, {whiteBalanceMode: 'continuous'}]
            }
        };
        navigator.mediaDevices.getUserMedia(camConstraints)
                .catch(() => navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: 'environment',
                        width: {ideal: 1920},
                        height: {ideal: 1080}
                    }
                }))
                .then(stream => {
                    camStream = stream;
                    const v = document.getElementById('cam-video');
                    v.srcObject = stream;
                    document.getElementById('cam-overlay').style.display = 'flex';
                    v.addEventListener('loadedmetadata', () => {
                        document.getElementById('cam-res').textContent = v.videoWidth + 'x' + v.videoHeight;
                        startCamQuality();
                    }, {once: true});
                })
                .catch(() => document.getElementById('file-cam').click());
    }

    function snapPhoto() {
        const v = document.getElementById('cam-video'), c = document.getElementById('cam-canvas');
        c.width = v.videoWidth;
        c.height = v.videoHeight;
        const ctx = c.getContext('2d');
        ctx.drawImage(v, 0, 0);
        stopCamQuality();
        closeCamera();
        c.toBlob(blob => {
            const f = new File([blob], 'photo.jpg', {type: 'image/jpeg'});
            const dt = new DataTransfer();
            dt.items.add(f);
            const inp = document.getElementById('file-upload');
            inp.files = dt.files;
            handleImage(inp);
        }, 'image/jpeg', 0.97);
    }

    function closeCamera() {
        stopCamQuality();
        if (camStream) {
            camStream.getTracks().forEach(t => t.stop());
            camStream = null;
        }
        document.getElementById('cam-overlay').style.display = 'none';
    }

    let _camQualInterval = null;

    function startCamQuality() {
        _camQualInterval = setInterval(() => {
            const v = document.getElementById('cam-video');
            if (!v || !v.videoWidth) return;
            // Sample center 60% of video frame for quality check
            const sw = Math.round(v.videoWidth * 0.6), sh = Math.round(v.videoHeight * 0.6);
            const ox = Math.round((v.videoWidth - sw) / 2), oy = Math.round((v.videoHeight - sh) / 2);
            const tc = document.createElement('canvas');
            tc.width = sw;
            tc.height = sh;
            tc.getContext('2d').drawImage(v, ox, oy, sw, sh, 0, 0, sw, sh);
            const d = tc.getContext('2d').getImageData(0, 0, sw, sh).data;
            let eS = 0, bSum = 0;
            for (let y = 1; y < sh - 1; y++) for (let x = 1; x < sw - 1; x++) {
                const i = (y * sw + x) * 4;
                eS += Math.abs(d[i - 4] - d[i + 4]) + Math.abs(d[i - sw * 4] - d[i + sw * 4]);
                bSum += (d[i] + d[i + 1] + d[i + 2]) / 3;
            }
            const brightness = bSum / (sw * sh);
            const diagF = Math.sqrt(sw * sw + sh * sh) / 400;
            const sharp = Math.round(Math.min(100, ((eS / (sw * sh)) / Math.max(0.5, diagF) / 30) * 100));
            const bright = brightness >= 60 && brightness <= 210;
            const el = document.getElementById('cam-steady');
            if (!el) return;
            if (sharp >= 65 && bright) {
                el.textContent = '✅ Hold steady';
                el.className = 'cam-steady good';
            } else if (sharp >= 40) {
                el.textContent = '⚠️ Almost — hold still';
                el.className = 'cam-steady';
            } else if (brightness < 50) {
                el.textContent = '🌑 Too dark';
                el.className = 'cam-steady';
            } else if (brightness > 220) {
                el.textContent = '✨ Too bright / glare';
                el.className = 'cam-steady';
            } else {
                el.textContent = '🌫️ Move closer / focus';
                el.className = 'cam-steady';
            }
        }, 500);
    }

    function stopCamQuality() {
        if (_camQualInterval) {
            clearInterval(_camQualInterval);
            _camQualInterval = null;
        }
    }

    function handleImage(inp) {
        const f = inp.files[0];
        if (!f) return;
        croppedBlob = null;
        verifyState = null;
        currentResult = null;
        currentScanId = null;
        const img = document.getElementById('preview');
        img.src = URL.createObjectURL(f);
        document.getElementById('scanner-guide').style.display = 'none';
        document.getElementById('scanner-preview').style.display = 'block';
        document.getElementById('scan-result-panel').style.display = 'none';
        document.getElementById('scanner-spinner').style.display = 'none';
        document.getElementById('crop-section-s').style.display = 'none';
        document.getElementById('scanner-action-bar').style.display = 'flex';
        img.onload = () => {
            resetZoom();
            initZoom();
            initBox();
            updateQ();
            updateLive();
        };
    }

    function resetScanner() {
        croppedBlob = null;
        currentResult = null;
        currentScanId = null;
        verifyState = null;
        resetZoom();
        document.getElementById('scanner-guide').style.display = 'block';
        ['scanner-preview', 'scan-result-panel', 'scanner-spinner'].forEach(id => {
            document.getElementById(id).style.display = 'none';
        });
        ['file-cam', 'file-upload'].forEach(id => {
            document.getElementById(id).value = '';
        });
        ['meta-batch', 'meta-notes'].forEach(id => {
            document.getElementById(id).value = '';
        });
        document.getElementById('save-status').style.display = 'none';
        document.getElementById('vchip-ok').className = 'verify-chip';
        document.getElementById('vchip-no').className = 'verify-chip';
        document.getElementById('correction-row').style.display = 'none';
    }

    // ============================================================
    //  PINCH TO ZOOM
    // ============================================================
    let _pz = {scale: 1, lastDist: 0, lastTap: 0};

    function _getTouchDist(e) {
        const t = e.touches;
        return Math.hypot(t[0].clientX - t[1].clientX, t[0].clientY - t[1].clientY);
    }

    function _applyZoom(s) {
        _pz.scale = Math.max(1, Math.min(4, s));
        const img = document.getElementById('preview');
        if (!img) return;
        img.style.transform = _pz.scale === 1 ? '' : 'scale(' + _pz.scale + ')';
        const zi = document.getElementById('zoom-indicator');
        if (zi) {
            zi.textContent = _pz.scale.toFixed(1) + 'x';
            zi.classList.add('show');
            clearTimeout(_pz._zt);
            _pz._zt = setTimeout(() => zi.classList.remove('show'), 1200);
        }
        if (_pz.scale === 1) {
            const pw = document.getElementById('preview-wrap');
            pw.style.overflowX = 'hidden';
        } else {
            const pw = document.getElementById('preview-wrap');
            pw.style.overflowX = 'auto';
        }
    }

    function initZoom() {
        const pw = document.getElementById('preview-wrap');
        if (!pw) return;
        pw.addEventListener('touchstart', function (e) {
            if (e.touches.length === 2) {
                e.preventDefault();
                _pz.lastDist = _getTouchDist(e);
            } else if (e.touches.length === 1) {
                const now = Date.now();
                if (now - _pz.lastTap < 300) {
                    e.preventDefault();
                    _applyZoom(_pz.scale === 1 ? 2.5 : 1);
                }
                _pz.lastTap = now;
            }
        }, {passive: false});
        pw.addEventListener('touchmove', function (e) {
            if (e.touches.length === 2) {
                e.preventDefault();
                const d = _getTouchDist(e);
                _applyZoom(_pz.scale * (d / _pz.lastDist));
                _pz.lastDist = d;
            }
        }, {passive: false});
        // Mouse wheel zoom for desktop/tablet
        pw.addEventListener('wheel', function (e) {
            e.preventDefault();
            _applyZoom(_pz.scale * (e.deltaY < 0 ? 1.1 : 0.9));
        }, {passive: false});
    }

    function resetZoom() {
        _applyZoom(1);
    }

    // ============================================================
    //  BOX / FOCUS REGION
    // ============================================================
    function applyBox() {
        const f = document.getElementById('focus-box');
        f.style.left = box.x + 'px';
        f.style.top = box.y + 'px';
        f.style.width = box.w + 'px';
        f.style.height = box.h + 'px';
    }

    function getImgLayout() {
        const img = document.getElementById('preview'), dW = img.clientWidth, dH = img.clientHeight;
        const natAR = img.naturalWidth / img.naturalHeight, dAR = dW / dH;
        let iW, iH, iX, iY;
        if (natAR < dAR) {
            iH = dH;
            iW = iH * natAR;
            iX = (dW - iW) / 2;
            iY = 0;
        } else {
            iW = dW;
            iH = iW / natAR;
            iX = 0;
            iY = (dH - iH) / 2;
        }
        return {iW, iH, iX, iY, sx: img.naturalWidth / iW, sy: img.naturalHeight / iH};
    }

    function initBox() {
        const {iW, iH, iX, iY} = getImgLayout();
        const s = Math.round(Math.min(iW, iH) * 0.50);
        box = {x: Math.round(iX + (iW - s) / 2), y: Math.round(iY + (iH - s) / 2), w: s, h: s};
        applyBox();
        setTimeout(autoFindFiber, 100);
    }

    function autoFindFiber() {
        const img = document.getElementById('preview');
        if (!img.naturalWidth) return;
        const NW = img.naturalWidth, NH = img.naturalHeight;
        const scale = Math.min(200 / NW, 200 / NH, 1);
        const sw = Math.round(NW * scale), sh = Math.round(NH * scale);
        const oc = document.createElement('canvas');
        oc.width = sw;
        oc.height = sh;
        oc.getContext('2d').drawImage(img, 0, 0, NW, NH, 0, 0, sw, sh);
        const data = oc.getContext('2d').getImageData(0, 0, sw, sh).data;
        const good = new Uint8Array(sw * sh);
        for (let i = 0; i < sw * sh; i++) {
            const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
            const brightness = (r + g + b) / 3, maxC = Math.max(r, g, b), minC = Math.min(r, g, b);
            if (brightness > 40 && brightness < 220 && maxC > 0 && (maxC - minC) / maxC > 0.08) good[i] = 1;
        }
        const integral = new Float32Array((sw + 1) * (sh + 1));
        for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) {
            integral[(y + 1) * (sw + 1) + (x + 1)] = good[y * sw + x] + integral[y * (sw + 1) + (x + 1)] + integral[(y + 1) * (sw + 1) + x] - integral[y * (sw + 1) + x];
        }

        function areaSum(x0, y0, x1, y1) {
            return integral[(y1 + 1) * (sw + 1) + (x1 + 1)] - integral[y0 * (sw + 1) + (x1 + 1)] - integral[(y1 + 1) * (sw + 1) + x0] + integral[y0 * (sw + 1) + x0];
        }

        let bestScore = -1, bestX = 0, bestY = 0, bestW = sw, bestH = sh;
        const stepX = Math.max(1, Math.round(sw * 0.05)), stepY = Math.max(1, Math.round(sh * 0.05));
        for (const frac of [0.75, 0.65, 0.55]) {
            const bw = Math.round(sw * frac), bh = Math.round(sh * frac);
            if (bw < sw * 0.25 || bh < sh * 0.25) continue;
            for (let y = 0; y + bh <= sh; y += stepY) for (let x = 0; x + bw <= sw; x += stepX) {
                const density = areaSum(x, y, x + bw - 1, y + bh - 1) / (bw * bh);
                if (density > bestScore) {
                    bestScore = density;
                    bestX = x;
                    bestY = y;
                    bestW = bw;
                    bestH = bh;
                }
            }
        }
        if (bestScore < 0.45) return;  // lowered from 0.70 — works for farm/mixed lighting photos
        const {iW, iH, iX, iY} = getImgLayout();
        const d = iW / sw;
        box = {
            x: Math.round(iX + bestX * d),
            y: Math.round(iY + bestY * d),
            w: Math.round(bestW * d),
            h: Math.round(bestH * d)
        };
        clampBox();
        applyBox();
        onChange();
    }

    function clampBox() {
        const pw = document.getElementById('preview-wrap'), W = pw.clientWidth, H = pw.clientHeight;
        // No maximum size limit — grader should be able to cover the whole image
        box.w = Math.max(60, box.w);
        box.h = Math.max(60, box.h);
        // Keep box within the preview container
        box.x = Math.max(0, Math.min(box.x, W - box.w));
        box.y = Math.max(0, Math.min(box.y, H - box.h));
        // If box extends beyond right/bottom edge, shrink it rather than move it
        if (box.x + box.w > W) box.w = W - box.x;
        if (box.y + box.h > H) box.h = H - box.y;
    }

    function pp(e, w) {
        const r = w.getBoundingClientRect(), t = e.touches ? e.touches[0] : e;
        return {x: t.clientX - r.left, y: t.clientY - r.top};
    }

    let _longPressTimer = null;
    let _longPressTarget = null;

    function onWD(e) {
        const pw = document.getElementById('preview-wrap');
        if (e.target !== document.getElementById('preview')) return;
        const p = pp(e, pw);
        const inBox = (p.x >= box.x - 10 && p.x <= box.x + box.w + 10 &&
                       p.y >= box.y - 10 && p.y <= box.y + box.h + 10);
        if (inBox) {
            dragging = 'move';
            // Store initial box position and pointer position separately
            dragStart = {bx: box.x, by: box.y, bw: box.w, bh: box.h, px: p.x, py: p.y};
            document.getElementById('focus-box').classList.add('dragging');
            e.preventDefault();
        } else {
            _longPressTarget = p;
            _longPressTimer = setTimeout(() => {
                const fb = document.getElementById('focus-box');
                fb.classList.add('long-press-flash');
                setTimeout(() => fb.classList.remove('long-press-flash'), 300);
                box.x = Math.round(p.x - box.w / 2);
                box.y = Math.round(p.y - box.h / 2);
                clampBox();
                applyBox();
                onChange();
                dragging = 'move';
                dragStart = {bx: box.x, by: box.y, bw: box.w, bh: box.h, px: p.x, py: p.y};
                fb.classList.add('dragging');
            }, 400);
        }
    }

    function onBD(e) {
        e.stopPropagation();
        const p = pp(e, document.getElementById('preview-wrap'));
        dragging = e.currentTarget.dataset.corner || 'move';
        // Store initial box state and pointer position separately
        dragStart = {bx: box.x, by: box.y, bw: box.w, bh: box.h, px: p.x, py: p.y};
        document.getElementById('focus-box').classList.add('dragging');
        e.preventDefault();
    }

    function onMv(e) {
        if (_longPressTimer) { clearTimeout(_longPressTimer); _longPressTimer = null; }
        if (!dragging || !dragStart) return;
        e.preventDefault();
        const p = pp(e, document.getElementById('preview-wrap'));
        const dx = p.x - dragStart.px;
        const dy = p.y - dragStart.py;
        if (dragging === 'move') {
            box.x = dragStart.bx + dx;
            box.y = dragStart.by + dy;
        } else if (dragging === 'br') {
            box.w = Math.max(60, dragStart.bw + dx);
            box.h = Math.max(60, dragStart.bh + dy);
        } else if (dragging === 'bl') {
            const nw = Math.max(60, dragStart.bw - dx);
            box.x = dragStart.bx + (dragStart.bw - nw);
            box.w = nw;
            box.h = Math.max(60, dragStart.bh + dy);
        } else if (dragging === 'tr') {
            box.w = Math.max(60, dragStart.bw + dx);
            const nh = Math.max(60, dragStart.bh - dy);
            box.y = dragStart.by + (dragStart.bh - nh);
            box.h = nh;
        } else if (dragging === 'tl') {
            const nw = Math.max(60, dragStart.bw - dx);
            const nh = Math.max(60, dragStart.bh - dy);
            box.x = dragStart.bx + (dragStart.bw - nw);
            box.y = dragStart.by + (dragStart.bh - nh);
            box.w = nw;
            box.h = nh;
        } else if (dragging === 'et') {
            const nh = Math.max(60, dragStart.bh - dy);
            box.y = dragStart.by + (dragStart.bh - nh);
            box.h = nh;
        } else if (dragging === 'eb') {
            box.h = Math.max(60, dragStart.bh + dy);
        } else if (dragging === 'el') {
            const nw = Math.max(60, dragStart.bw - dx);
            box.x = dragStart.bx + (dragStart.bw - nw);
            box.w = nw;
        } else if (dragging === 'er') {
            box.w = Math.max(60, dragStart.bw + dx);
        }
        clampBox();
        applyBox();
        // Throttled live update while dragging
        if (!onMv._t) onMv._t = setTimeout(() => { onMv._t = null; onChange(); }, 60);
    }

    function onUp() {
        // Cancel any pending long-press
        if (_longPressTimer) { clearTimeout(_longPressTimer); _longPressTimer = null; }
        if (dragging) {
            dragging = null;
            dragStart = null;
            document.getElementById('focus-box').classList.remove('dragging');
            onChange();
        }
    }

    function onChange() {
        clearTimeout(liveTimer);
        liveTimer = setTimeout(() => {
            updateQ();
            updateLive();
        }, 80);
    }

    function getCrop() {
        const img = document.getElementById('preview');
        if (!img.naturalWidth) return null;
        const {iX, iY, sx, sy} = getImgLayout();
        const bx = Math.max(0, Math.round((box.x - iX) * sx)), by = Math.max(0, Math.round((box.y - iY) * sy));
        const bw = Math.min(img.naturalWidth - bx, Math.round(box.w * sx)),
                bh = Math.min(img.naturalHeight - by, Math.round(box.h * sy));
        if (bw < 80 || bh < 80) return null;  // minimum 80x80px natural pixels
        const c = document.createElement('canvas');
        c.width = bw;
        c.height = bh;
        const ctx = c.getContext('2d');
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(img, bx, by, bw, bh, 0, 0, bw, bh);
        return {canvas: c, bx, by, bw, bh, W: img.naturalWidth, H: img.naturalHeight};
    }

    function updateLive() {
        const r = getCrop();
        if (!r) return;
        const {canvas: c, bw, bh} = r;
        const data = c.getContext('2d').getImageData(0, 0, bw, bh).data;
        let rv = 0, gv = 0, bv = 0, n = 0;
        const x0 = Math.floor(bw * .25), x1 = Math.floor(bw * .75), y0 = Math.floor(bh * .25),
                y1 = Math.floor(bh * .75);
        for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
            const i = (y * bw + x) * 4;
            rv += data[i];
            gv += data[i + 1];
            bv += data[i + 2];
            n++;
        }
        if (!n) return;
        rv = Math.round(rv / n);
        gv = Math.round(gv / n);
        bv = Math.round(bv / n);
        const h = `#${rv.toString(16).padStart(2,'0')}${gv.toString(16).padStart(2,'0')}${bv.toString(16).padStart(2,'0')}`;
        document.getElementById('live-sw').style.background = h;
        document.getElementById('live-hex').textContent = h.toUpperCase();
        document.getElementById('color-swatch-big').style.background = h;
        document.getElementById('color-rgb-display').textContent = `RGB ${rv} ${gv} ${bv}`;
        const lApprox = Math.round(0.2126 * (rv / 255) * 100 + 0.7152 * (gv / 255) * 100 + 0.0722 * (bv / 255) * 100);
        document.getElementById('color-lab-display').textContent = `L* ${lApprox} a* — b* —`;
        let darkPx = 0, glarePx = 0, total = data.length / 4;
        for (let i = 0; i < data.length; i += 4) {
            const pb = (data[i] + data[i + 1] + data[i + 2]) / 3;
            if (pb < 50) darkPx++;
            if (pb > 210) glarePx++;
        }
        const hint = document.getElementById('live-hint');
        const darkR = darkPx / total, glareR = glarePx / total;
        if (darkR > 0.15) {
            hint.textContent = '⚫ ' + Math.round(darkR * 100) + '% dark — move box!';
            hint.style.color = '#f05060';
        } else if (glareR > 0.15) {
            hint.textContent = '✨ ' + Math.round(glareR * 100) + '% glare — reposition!';
            hint.style.color = '#e09000';
        } else {
            hint.textContent = 'Live avg · avoid hole & glare';
            hint.style.color = 'rgba(255,255,255,.5)';
        }
    }

    function bcol(p) {
        return p >= 70 ? '#1a8c45' : p >= 40 ? '#c47c00' : '#c0001a';
    }

    function updateQ() {
        const r = getCrop();
        if (!r) return;
        const {canvas: c, bw, bh, W, H} = r;
        const data = c.getContext('2d').getImageData(0, 0, bw, bh).data;
        let bSum = 0;
        for (let i = 0; i < data.length; i += 4) bSum += (data[i] + data[i + 1] + data[i + 2]) / 3;
        const brightness = bSum / (bw * bh);
        let eS = 0;
        for (let y = 1; y < bh - 1; y++) for (let x = 1; x < bw - 1; x++) {
            const i = (y * bw + x) * 4;
            eS += Math.abs(data[i - 4] - data[i + 4]) + Math.abs(data[i - bw * 4] - data[i + bw * 4]);
        }
        const rawSharp = eS / (bw * bh);
        // Normalize sharpness by resolution: a sharp 300px crop should score same as sharp 1500px crop
        // Expected Laplacian response scales with pixel density, so normalize against image diagonal
        const diagFactor = Math.sqrt(bw * bw + bh * bh) / 400;
        const normSharp = rawSharp / Math.max(0.5, diagFactor);
        const cov = bw * bh / (W * H) * 100;
        let bp = brightness >= 60 && brightness <= 200 ? 100 : brightness >= 30 ? 40 + (brightness - 30) * 2 : 10;
        bp = Math.round(Math.min(100, Math.max(0, bp)));
        const sp = Math.round(Math.min(100, (normSharp / 30) * 100));
        const szp = Math.round(Math.min(100, cov / 20 * 100));
        document.getElementById('qb-bright').style.cssText = 'width:' + bp + '%;background:' + bcol(bp);
        document.getElementById('qb-sharp').style.cssText = 'width:' + sp + '%;background:' + bcol(sp);
        document.getElementById('qb-size').style.cssText = 'width:' + szp + '%;background:' + bcol(szp);
        document.getElementById('qv-bright').textContent = Math.round(brightness) + '/255';
        document.getElementById('qv-sharp').textContent = sp + '%';
        document.getElementById('qv-size').textContent = cov.toFixed(1) + '%';
        document.getElementById('qi-bright').textContent = bp >= 70 ? '☀️' : bp >= 40 ? '⚠️' : '🌑';
        document.getElementById('qi-sharp').textContent = sp >= 70 ? '🔍' : sp >= 40 ? '⚠️' : '🌫️';
        document.getElementById('qi-size').textContent = szp >= 70 ? '📐' : szp >= 40 ? '⚠️' : '📦';
        qualIssues = [];
        if (bp < 40) qualIssues.push('poor lighting');
        if (sp < 40) qualIssues.push('blurry image');
        if (cov < 10) qualIssues.push('box too small');
        // Live crop thumbnail
        const thumbEl = document.getElementById('live-crop-thumb');
        const wrapEl = document.getElementById('live-crop-wrap');
        if (thumbEl && wrapEl) {
            wrapEl.classList.add('show');
            const tc = thumbEl.getContext('2d');
            tc.drawImage(c, 0, 0, bw, bh, 0, 0, 54, 54);
            document.getElementById('live-crop-dims').textContent = bw + 'x' + bh + ' px';
            const warn = document.getElementById('live-crop-warn');
            if (qualIssues.length > 0) warn.textContent = '⚠ ' + qualIssues.join(' · ');
            else warn.textContent = '';
        }
    }

    function showCropS() {
        const r = getCrop();
        if (!r) return;
        const {canvas, bw, bh, W, H} = r;
        applyWbToCanvas(canvas);
        const cc = document.getElementById('crop-canvas-s');
        cc.width = bw;
        cc.height = bh;
        cc.getContext('2d').drawImage(canvas, 0, 0);
        document.getElementById('cs-w').textContent = bw;
        document.getElementById('cs-h').textContent = bh;
        document.getElementById('cs-pct').textContent = (bw * bh / (W * H) * 100).toFixed(1) + '%';
        canvas.toBlob(b => {
            croppedBlob = b;
        }, 'image/jpeg', .97);
        document.getElementById('scanner-action-bar').style.display = 'none';
        document.getElementById('crop-section-s').style.display = 'block';
        document.getElementById('crop-section-s').scrollIntoView({behavior: 'smooth', block: 'nearest'});
    }

    function hideCropS() {
        document.getElementById('crop-section-s').style.display = 'none';
        document.getElementById('scanner-action-bar').style.display = 'flex';
        croppedBlob = null;
    }

    function checkThenSubmit() {
        const r = getCrop();
        if (!r) { showModal('No image loaded. Please take or upload a photo first.'); return; }

        // ── Pre-flight rejection checks ───────────────────────────────────
        const pixels = r.canvas.getContext('2d').getImageData(0, 0, r.bw, r.bh).data;
        let lumSum = 0, darkPx = 0, brightPx = 0, total = r.bw * r.bh;
        for (let i = 0; i < pixels.length; i += 4) {
            const lum = (pixels[i] * 0.299 + pixels[i+1] * 0.587 + pixels[i+2] * 0.114);
            lumSum += lum;
            if (lum < 40) darkPx++;
            if (lum > 215) brightPx++;
        }
        const avgLum = lumSum / total;
        const darkRatio = darkPx / total;
        const brightRatio = brightPx / total;
        const cropPct = r.bw * r.bh / (r.W * r.H) * 100;

        // Hard reject — show clear message, no "analyze anyway" option
        if (cropPct < 2) {
            showModal('📐 Box too small. Drag the corners to cover more fiber area.', false); return;
        }
        if (avgLum < 25) {
            showModal('🌑 Too dark — almost black. Improve lighting or move to a brighter spot.', false); return;
        }
        if (brightRatio > 0.60) {
            showModal('✨ Too much glare or overexposure. Move away from direct light or shaded area.', false); return;
        }

        // Soft warn — show message but allow "analyze anyway"
        const warns = [];
        if (avgLum < 50) warns.push('⚠️ Lighting is dim — results may be inaccurate');
        if (darkRatio > 0.40) warns.push('⚫ ' + Math.round(darkRatio*100) + '% dark pixels — try isolating a single fiber strand');
        if (brightRatio > 0.25) warns.push('✨ ' + Math.round(brightRatio*100) + '% glare pixels — reposition to avoid shine');
        if (cropPct < 10) warns.push('📐 Crop box is small — cover more fiber for better accuracy');
        if (qualIssues.includes('blurry image')) warns.push('🌫️ Image appears blurry — hold steady when capturing');

        if (warns.length > 0) {
            showModal(warns.join(' · '), true); return;
        }
        doSubmit();
    }

    function showModal(msg, allowForce = true) {
        // Replace newlines with <br> for multi-line messages
        document.getElementById('qm-body').innerHTML = msg.replace(/\n/g, '<br>');
        const forceBtn = document.querySelector('#quality-modal .btn.danger');
        if (forceBtn) forceBtn.style.display = allowForce ? '' : 'none';
        document.getElementById('quality-modal').style.display = 'flex';
    }

    function closeModal() {
        document.getElementById('quality-modal').style.display = 'none';
    }

    function forceSubmit() {
        closeModal();
        doSubmit();
    }

    const STEPS = ['Segmenting fiber pixels…', 'Applying white balance correction…', 'Extracting dominant color…', 'Running MLP-A + MLP-B…', 'Running MLP-C + MLP-D…', 'Matching RHS Delta-E…'];
    let spinI = null;

    function startSpin() {
        let i = 0;
        document.getElementById('spin-step').textContent = STEPS[0];
        spinI = setInterval(() => {
            i = (i + 1) % STEPS.length;
            document.getElementById('spin-step').textContent = STEPS[i];
        }, 800);
    }

    function stopSpin() {
        clearInterval(spinI);
    }

    async function doSubmit() {
        if (!croppedBlob) {
            const r = getCrop();
            if (!r) return;
            r.canvas.toBlob(async b => {
                croppedBlob = b;
                await _send();
            }, 'image/jpeg', .97);
        } else {
            await _send();
        }
    }

    async function _send() {
        document.getElementById('crop-section-s').style.display = 'none';
        document.getElementById('scanner-action-bar').style.display = 'none';
        document.getElementById('scanner-spinner').style.display = 'block';
        document.getElementById('scan-result-panel').style.display = 'none';
        startSpin();
        try {
            const fd = new FormData();
            fd.append('image', croppedBlob, 'crop.jpg');
            if (wbProfile) {
                fd.append('wb_r', wbProfile.r);
                fd.append('wb_g', wbProfile.g);
                fd.append('wb_b', wbProfile.b);
            }
            const res = await fetch('/predict', {method: 'POST', body: fd});
            const d = await res.json();
            stopSpin();
            document.getElementById('scanner-spinner').style.display = 'none';
            if (d.error) {
                showError('Analysis failed', d.error);
                return;
            }
            renderScanResult(d);
        } catch (e) {
            stopSpin();
            document.getElementById('scanner-spinner').style.display = 'none';
            showError('Connection error', e.message);
        }
    }

    function renderScanResult(d) {
        currentResult = d;
        verifyState = null;
        document.getElementById('sib-grader').textContent = currentUser ? currentUser.username : '—';
        document.getElementById('sib-location').textContent = getFullLocation() || '⚠️ Not set';
        document.getElementById('sib-location').style.color = getFullLocation() ? 'var(--text)' : 'var(--warn)';

        // ── Preflight / confidence banner ────────────────────────────────
        const existing = document.getElementById('preflight-banner');
        if (existing) existing.remove();
        const tips = [];
        if (d.cast_warning && !d.wb_applied)
            tips.push({ icon: '🎨', text: d.cast_warning + ' — white balance was not fully corrected. Rescan under neutral daylight for best accuracy.' });
        if (d.delta_e > 8 || d.verdict === 'LIKELY MATCH' || d.verdict === 'UNCERTAIN')
            tips.push({ icon: '🔁', text: 'Low confidence (ΔE ' + d.delta_e.toFixed(1) + '). Try isolating a single fiber strand on white paper background.' });
        if (d.seg_coverage < 25)
            tips.push({ icon: '📐', text: 'Low fiber coverage (' + d.seg_coverage.toFixed(0) + '%). Get closer or expand the crop box over more fiber.' });
        if (tips.length > 0) {
            const banner = document.createElement('div');
            banner.id = 'preflight-banner';
            banner.style.cssText = 'margin:0 0 10px;background:var(--warn-bg);border:1px solid #f0d580;border-radius:10px;padding:10px 12px;';
            banner.innerHTML = tips.map(t =>
                `<div style="display:flex;gap:8px;align-items:flex-start;margin-bottom:${tips.length>1?'6px':'0'};">
                   <span style="font-size:1rem;flex-shrink:0;">${t.icon}</span>
                   <span style="font-size:.72rem;color:var(--warn);line-height:1.4;">${t.text}</span>
                 </div>`
            ).join('');
            const panel = document.querySelector('.scan-result-panel');
            if (panel) panel.insertBefore(banner, panel.firstChild);
        }
        document.getElementById('res-swatch-scan').style.background = d.dominant_hex;
        document.getElementById('res-swatch-ref').style.background = d.matched_hex;
        document.getElementById('res-rhs-code').textContent = d.rhs_code;
        const vchip = document.getElementById('res-verdict');
        vchip.textContent = d.verdict;
        vchip.style.cssText = `background:${d.verdict_color}18;color:${d.verdict_color};border:1px solid ${d.verdict_color}44;`;
        document.getElementById('res-de').textContent = d.delta_e.toFixed(2);
        document.getElementById('res-score').textContent = d.match_score.toFixed(1) + '%';
        document.getElementById('res-hex').textContent = d.dominant_hex.toUpperCase();
        document.getElementById('live-chart-match').textContent = d.rhs_code;
        // Tie banner — shown when top 2 candidates are within ΔE 5
        const tieBanner = d.is_tie && d.tie_code ? `
    <div class="tie-banner">
      <div class="tie-banner-icon">🔶</div>
      <div class="tie-banner-body">
        <div class="tie-banner-title">Too close to auto-select</div>
        <div class="tie-banner-codes">
          <span class="tie-swatch" style="background:${d.matched_hex||'#ccc'};"></span>
          ${d.rhs_code}
          <span style="color:#aaa;font-size:.8rem;">or</span>
          <span class="tie-swatch" style="background:${d.tie_hex||'#ccc'};"></span>
          ${d.tie_code}
        </div>
        <div class="tie-banner-hint">Colors are within ΔE 5 — use the physical card to confirm the grade.</div>
      </div>
    </div>` : '';
        document.getElementById('top5-list-s').innerHTML = tieBanner + d.top_5.map((t, i) => `
    <div class="top5-item${i===0?' selected':''}" id="top5-item-${t.rhs_code}" onclick="selectTop5('${t.rhs_code}')">
      <div style="width:32px;height:32px;border-radius:6px;background:${t.hex};border:1px solid var(--border);flex-shrink:0;"></div>
      <div style="flex:1;"><div style="font-family:IBM Plex Mono,monospace;font-size:.88rem;font-weight:600;">${t.rhs_code}${i===0?` <span style="font-size:.58rem;color:${d.is_tie?'#f5a623':'var(--accent)'};font-weight:700;">${d.is_tie?'TIED':'TOP'}</span>`
    :
        ''
    }${d.is_tie&&i===1?` <span style="font-size:.58rem;color:#f5a623;font-weight:700;">TIED</span>`:''}</div>
    <div style="font-size:.63rem;color:${t.de_color};">ΔE ${t.delta_e} · ${t.de_label}</div>
    </div>
    <div style="text-align:right;">
        <div style="font-size:.82rem;font-weight:700;color:${i===0?'var(--accent)':'var(--sub)'};">${t.match_score.toFixed(1)}%</div>
        <div class="top5-tap-hint">${i===0?(d.is_tie?'tap to confirm':'auto-selected'):d.is_tie&&i===1?'tap to confirm':'tap to select'}</div>
    </div>
    </div>
    `).join('');
  setVerify(true);document.getElementById('save-status').style.display='none';
  document.getElementById('scanner-preview').style.display='none';document.getElementById('scan-result-panel').style.display='block';document.getElementById('scan-result-panel').scrollIntoView({behavior:'smooth',block:'start'});
  sessionScanCount++;document.getElementById('sb-scan-count').textContent=sessionScanCount;
  const cnt=parseInt(document.getElementById('topbar-scan-count').textContent)||0;document.getElementById('topbar-scan-count').textContent=(cnt+1)+' scans';
  const sessionEntry={
    _session_id:'sess_'+Date.now()+'_'+Math.random().toString(36).slice(2,7),_unsaved:true,rhs_grade:d.rhs_code,delta_e:d.delta_e,match_score:d.match_score,
    dominant_hex:d.dominant_hex,matched_hex:d.matched_hex,verdict:d.verdict,verdict_color:d.verdict_color,dominant_rgb:d.dominant_rgb,dominant_lab:d.dominant_lab,top_5:d.top_5,
    seg_found:d.seg_found,seg_coverage:d.seg_coverage,wb_applied:d.wb_applied,cast_label:d.cast_label,cast_warning:d.cast_warning,
    thumbnail_b64:document.getElementById('crop-canvas-s').toDataURL('image/jpeg',0.6),
    scanned_at:new Date().toISOString(),verified:0,correction:'',batch_id:'',grader_notes:'',location:getFullLocation(),grader_name:currentUser?currentUser.username:'',
  };
  scanHistory.unshift(sessionEntry);_storeAndKey(sessionEntry);currentResult._session_id=sessionEntry._session_id;
}
function selectTop5(code){
  if(currentResult)currentResult.top_5.forEach(t=>{const el=document.getElementById('top5-item-'+t.rhs_code);if(el)el.classList.remove('selected');});
  const el=document.getElementById('top5-item-'+code);if(el)el.classList.add('selected');
  if(currentResult&&code===currentResult.rhs_code){setVerify(true);}else{setVerify(false);document.getElementById('correction-input').value=code;}
}
function scanAgain(){resetScanner();switchScreen('scanner');}
function setVerify(isCorrect){
  verifyState=isCorrect;
  document.getElementById('vchip-ok').className='verify-chip'+(isCorrect?' selected-ok':'');
  document.getElementById('vchip-no').className='verify-chip'+(!isCorrect?' selected-no':'');
  document.getElementById('correction-row').style.display=isCorrect?'none':'block';
}

// ============================================================
//  SAVE
// ============================================================
async function saveScan(){
  if(!currentResult){showError('No result','Please analyze a fiber first.');return;}
  if(!getFullLocation())showLocToast();
  const btn=document.querySelector('[onclick="saveScan()"]');btn.disabled=true;btn.textContent='💾 Saving...';
  const payload={
    rhs_grade:currentResult.rhs_code,user_id:currentUser?currentUser.id:null,rhs_code:currentResult.rhs_code,
    delta_e:currentResult.delta_e,match_score:currentResult.match_score,
    rgb_r:currentResult.dominant_rgb?.R,rgb_g:currentResult.dominant_rgb?.G,rgb_b:currentResult.dominant_rgb?.B,
    lab_l:currentResult.dominant_lab?.L,lab_a:currentResult.dominant_lab?.a,lab_b:currentResult.dominant_lab?.b,
    dominant_hex:currentResult.dominant_hex,matched_hex:currentResult.matched_hex,verdict:currentResult.verdict,verdict_color:currentResult.verdict_color,
    batch_id:document.getElementById('meta-batch').value.trim(),grader_notes:document.getElementById('meta-notes').value.trim(),
    verified:verifyState===true?1:0,correction:verifyState===false?document.getElementById('correction-input').value.trim().toUpperCase():'',
    thumbnail_b64:document.getElementById('crop-canvas-s').toDataURL('image/jpeg',0.7),
    location:getFullLocation(),grader_name:currentUser?currentUser.username:'',
    top_5:currentResult.top_5,dominant_rgb:currentResult.dominant_rgb,dominant_lab:currentResult.dominant_lab,
    seg_found:currentResult.seg_found,seg_coverage:currentResult.seg_coverage,wb_applied:currentResult.wb_applied,
  };
  try{
    const r=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const d=await r.json();
    if(d.error){showError('Save failed',d.error);btn.disabled=false;btn.textContent='💾 Verify & Save';return;}
    currentScanId=d.scan_id;

    // ── Upload real crop photo for retraining (best-effort, non-blocking) ──
    // croppedBlob is the full-resolution crop from the scan — much better than
    // the compressed thumbnail_b64. Uploaded to Supabase Storage so the retrain
    // endpoint can fetch the real field photo instead of synthesizing a swatch.
    if(croppedBlob && d.scan_id){
      try{
        const photoFd=new FormData();
        photoFd.append('scan_id', d.scan_id);
        photoFd.append('image', croppedBlob, 'crop.jpg');
        fetch('/api/save-photo',{method:'POST',body:photoFd})
          .catch(()=>{}); // fire-and-forget — never blocks save
      }catch(_){}
    }

    const idx=scanHistory.findIndex(e=>e._session_id===currentResult._session_id);
    if(idx>=0)Object.assign(scanHistory[idx],{_unsaved:false,id:d.scan_id,verified:payload.verified,correction:payload.correction,batch_id:payload.batch_id,grader_notes:payload.grader_notes,location:payload.location,grader_name:payload.grader_name});
    const ss=document.getElementById('save-status');ss.style.display='block';
    ss.innerHTML='✅ Saved! ID: '+(payload.batch_id||d.scan_id.slice(0,8).toUpperCase())+'<br><button onclick="scanAgain()" style="margin-top:8px;width:100%;padding:12px;background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;border:none;border-radius:9px;font-size:.88rem;font-weight:800;cursor:pointer;font-family:Plus Jakarta Sans,sans-serif;">📷 Scan Again</button>';
    btn.disabled=false;btn.textContent='✅ Saved!';setTimeout(()=>{btn.textContent='💾 Verify & Save';},2500);
  }catch(e){showError('Save error',e.message);btn.disabled=false;btn.textContent='💾 Verify & Save';}
}
function showError(title,body){
  document.getElementById('err-title').textContent=title;document.getElementById('err-body').textContent=' '+body;
  const t=document.getElementById('err-toast');t.style.display='block';setTimeout(()=>{t.style.display='none';},4000);
}
(function(){
  const pw=document.getElementById('preview-wrap'),fb=document.getElementById('focus-box');

  // Prevent the image from scrolling/panning while any drag is active
  pw.addEventListener('touchstart', function(e) {
    // Only block scroll if this is a single-finger touch that starts a drag
    if (e.touches.length === 1 && (dragging || e.target === document.getElementById('preview'))) {
      // Will be handled by onWD — don't let it scroll
    }
  }, {passive: true});

  pw.addEventListener('mousedown',onWD);
  pw.addEventListener('touchstart',onWD,{passive:false});

  // The focus-box itself handles move — don't let it bubble to preview-wrap's onWD
  fb.addEventListener('mousedown', function(e) {
    // If clicking the box body (not a handle), treat as move
    if (e.target === fb || e.target === document.getElementById('focus-dot')) {
      onBD.call({dataset:{corner:'move'}}, e);
    } else {
      onBD.call(e.currentTarget, e);
    }
  });
  fb.addEventListener('touchstart', function(e) {
    if (e.target === fb || e.target === document.getElementById('focus-dot')) {
      onBD.call({dataset:{corner:'move'}}, e);
    } else {
      onBD.call(e.currentTarget, e);
    }
  }, {passive:false});

  // Wire all corner AND edge handles
  fb.querySelectorAll('.corner,.edge').forEach(el=>{
    el.addEventListener('mousedown', function(e) { e.stopPropagation(); onBD.call(this, e); });
    el.addEventListener('touchstart', function(e) { e.stopPropagation(); onBD.call(this, e); }, {passive:false});
  });

  document.addEventListener('mousemove',onMv);
  document.addEventListener('touchmove',onMv,{passive:false});
  document.addEventListener('mouseup',onUp);
  document.addEventListener('touchend',onUp);
})();

// ============================================================