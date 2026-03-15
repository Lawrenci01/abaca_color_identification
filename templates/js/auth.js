//  AUTH
// ============================================================
function checkSession(){
  try{
    const saved=localStorage.getItem('abaca_user');
    if(saved){
      currentUser=JSON.parse(saved);
      try{sessionLocation=sessionStorage.getItem('abaca_loc')||'';}catch(e){}
      showApp();
    }else{
      showLogin();
    }
  }catch(e){showLogin();}
}

function showLogin(){
  // Hide app chrome
  document.querySelector('.nav').style.display='none';
  document.querySelector('.topbar').style.display='none';
  const sb=document.getElementById('session-bar');if(sb)sb.style.display='none';

  // Deactivate all app screens
  document.querySelectorAll('#app .screen').forEach(s=>s.classList.remove('active'));

  // Show login overlay (position:fixed, z-index:900, display:flex)
  const lm=document.getElementById('login-mount');
  if(lm)lm.style.display='flex';
}

function isAdmin(){return currentUser&&currentUser.role==='admin';}

function showApp(){
  // Hide login overlay
  const lm=document.getElementById('login-mount');
  if(lm)lm.style.display='none';

  // Show app chrome
  document.querySelector('.topbar').style.display='flex';
  const sb=document.getElementById('session-bar');if(sb)sb.style.display='flex';

  if(isAdmin()){
    document.body.classList.add('is-admin');
    document.querySelector('.nav').style.display='flex';
    document.querySelector('.nav').innerHTML=`
    <div class="nav-item active" id="nav-admin-dashboard" onclick="switchAdminScreen('dashboard')">
        <div class="nav-icon">📊</div>
        <div class="nav-label">Overview</div>
    </div>
    <div class="nav-item" id="nav-admin-history" onclick="switchAdminScreen('history')">
        <div class="nav-icon">🕐</div>
        <div class="nav-label">History</div>
    </div>
    <div class="nav-item" id="nav-admin-users" onclick="switchAdminScreen('users')">
        <div class="nav-icon">👥</div>
        <div class="nav-label">Users</div>
    </div>
    <div class="nav-item" id="nav-admin-settings" onclick="switchAdminScreen('settings')">
        <div class="nav-icon">⚙️</div>
        <div class="nav-label">Settings</div>
    </div>`;
    document.getElementById('topbar-scan-count').innerHTML='<span class="admin-role-badge">ADMIN</span>';
    const el=document.getElementById('adm-settings-username');if(el)el.textContent=currentUser.username;
    switchAdminScreen('dashboard');
  }else{
    document.body.classList.remove('is-admin');
    document.querySelector('.nav').style.display='flex';
    document.querySelector('.nav').innerHTML=`
    <div class="nav-item active" id="nav-dashboard" onclick="switchScreen('dashboard')">
        <div class="nav-icon">📊</div>
        <div class="nav-label">Dashboard</div>
    </div>
    <div class="nav-item" id="nav-scanner" onclick="switchScreen('scanner')">
        <div class="nav-icon">📷</div>
        <div class="nav-label">Scanner</div>
    </div>
    <div class="nav-item" id="nav-history" onclick="switchScreen('history')">
        <div class="nav-icon">🕐</div>
        <div class="nav-label">History</div>
    </div>
    <div class="nav-item" id="nav-settings" onclick="switchScreen('settings')">
        <div class="nav-icon">⚙️</div>
        <div class="nav-label">Settings</div>
    </div>`;
    const inp=document.getElementById('sb-loc-input');if(inp)inp.value=sessionLocation;
    updateSessionBar();loadWbProfile();switchScreen('dashboard');
    if(!sessionLocation){showLocationSetup();}
  }
}

function doLogout(){
  if(!confirm('Sign out of '+currentUser.username+'?'))return;
  document.body.classList.remove('is-admin');
  currentUser=null;sessionLocation='';sessionScanCount=0;
  localStorage.removeItem('abaca_user');
  try{sessionStorage.removeItem('abaca_loc');}catch(e){}
  loginPin='';regPin='';adminPin='';
  resetPinDots('pd');resetPinDots('rpd');resetPinDots('apd');
  showLogin();
}

function switchLoginTab(tab){
  ['login','register','admin'].forEach(t=>{
    document.getElementById('ltab-'+t)?.classList.toggle('active',t===tab);
    document.getElementById('lform-'+t).style.display=t===tab?'block':'none';
  });
  ['li-err','reg-err','adm-err'].forEach(id=>{const el=document.getElementById(id);if(el)el.textContent='';});
  loginPin='';adminPin='';regPin='';
  resetPinDots('pd');resetPinDots('rpd');resetPinDots('apd');
}

function npPress(form,digit){
  if(form==='login'){
    if(loginPin.length>=4)return;loginPin+=digit;updatePinDots('pd',loginPin);
    if(loginPin.length===4){const u=document.getElementById('li-username').value.trim();if(u)setTimeout(()=>doLogin(),200);else document.getElementById('li-err').textContent='Enter your username first ↑';}
  }else if(form==='reg'){
    if(regPin.length>=4)return;regPin+=digit;updatePinDots('rpd',regPin);
  }else if(form==='admin'){
    if(adminPin.length>=4)return;adminPin+=digit;updatePinDots('apd',adminPin);
    if(adminPin.length===4){const u=document.getElementById('adm-login-username').value.trim();if(u)setTimeout(()=>doAdminLogin(),200);else document.getElementById('adm-err').textContent='Enter admin username first ↑';}
  }
}

function npDel(form){
  if(form==='login'){loginPin=loginPin.slice(0,-1);updatePinDots('pd',loginPin);}
  else if(form==='reg'){regPin=regPin.slice(0,-1);updatePinDots('rpd',regPin);}
  else if(form==='admin'){adminPin=adminPin.slice(0,-1);updatePinDots('apd',adminPin);}
}

function updatePinDots(prefix,pin){
  for(let i=0;i<4;i++){
    const d=document.getElementById(prefix+i);
    if(!d)continue;
    d.textContent=pin.length>i?'●':'';
    d.className='pin-dot'+(pin.length>i?' filled':pin.length===i?' active':'');
  }
}

function resetPinDots(prefix){
  for(let i=0;i<4;i++){
    const d=document.getElementById(prefix+i);
    if(d){d.textContent='';d.className='pin-dot';}
  }
}

async function doLogin(){
  const username=document.getElementById('li-username').value.trim();
  const err=document.getElementById('li-err');
  if(!username){err.textContent='Please enter your username';return;}
  if(loginPin.length<4){err.textContent='Please enter your 4-digit PIN';return;}
  err.textContent='Signing in…';
  try{
    const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username,pin:loginPin})});
    const d=await r.json();
    if(d.error){err.textContent=d.error;loginPin='';resetPinDots('pd');return;}
    currentUser={id:d.id,username:d.username};
    localStorage.setItem('abaca_user',JSON.stringify(currentUser));
    showApp();startHeartbeat();
  }catch(e){err.textContent='Connection error';loginPin='';resetPinDots('pd');}
}

async function doRegister(){
  const username=document.getElementById('reg-username').value.trim();
  const err=document.getElementById('reg-err');
  if(!username){err.textContent='Please enter a username';return;}
  if(username.length<3){err.textContent='Username must be at least 3 characters';return;}
  if(regPin.length<4){err.textContent='Please enter a 4-digit PIN';return;}
  err.textContent='Creating account…';
  try{
    const r=await fetch('/api/register',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username,pin:regPin})});
    const d=await r.json();
    if(d.error){err.textContent=d.error;regPin='';resetPinDots('rpd');return;}
    currentUser={id:d.id,username:d.username};
    localStorage.setItem('abaca_user',JSON.stringify(currentUser));
    showApp();startHeartbeat();
  }catch(e){err.textContent='Connection error';regPin='';resetPinDots('rpd');}
}

async function doAdminLogin(){
  const username=document.getElementById('adm-login-username').value.trim();
  const err=document.getElementById('adm-err');
  if(!username){err.textContent='Please enter your admin username';return;}
  if(adminPin.length<4){err.textContent='Please enter your 4-digit PIN';return;}
  err.textContent='Authenticating…';
  try{
    const r=await fetch('/api/admin/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username,pin:adminPin})});
    const d=await r.json();
    if(d.error){err.textContent=d.error;adminPin='';resetPinDots('apd');return;}
    currentUser={id:d.id,username:d.username,role:'admin'};
    localStorage.setItem('abaca_user',JSON.stringify(currentUser));
    showApp();
  }catch(e){err.textContent='Connection error';adminPin='';resetPinDots('apd');}
}

// ============================================================