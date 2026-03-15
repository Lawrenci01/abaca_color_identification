//  SUPABASE SYNC for db.py
// ============================================================
// NOTE: Your Supabase "users" table is in the image above.
// The db.py reads from env vars SUPABASE_URL and SUPABASE_KEY.
// Set these in your HF Space secrets panel.
// Admin login uses a SEPARATE "admins" table (seeded on first boot).
// The admin row in the Supabase "users" table (username: admin)
// is a regular user — NOT the admin login. The admin login
// goes through /api/admin/login which checks the "admins" table.

// ============================================================
//  INIT
// ============================================================
function showLocationSetup(){
  document.getElementById('screen-location-setup').style.display='flex';
  setTimeout(()=>document.getElementById('setup-location-input').focus(),300);
}
function confirmLocation(){
  const val=document.getElementById('setup-location-input').value.trim();
  if(!val){document.getElementById('setup-location-input').style.border='1px solid var(--danger)';return;}
  sessionLocation=val;
  try{sessionStorage.setItem('abaca_loc',val);}catch(e){}
  const inp=document.getElementById('sb-loc-input');if(inp)inp.value=val;
  updateSessionBar();
  document.getElementById('screen-location-setup').style.display='none';
}
function skipLocation(){
  document.getElementById('screen-location-setup').style.display='none';
}
checkSession();