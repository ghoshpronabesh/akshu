(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const state = {mode:'idle', voiceEnabled:false, muted:false, speaking:false, lastSpoken:new Set(), lastEventId:0, demoTimer:null, demoIndex:0, pollTimer:null};
  const demoScenes = [
    {title:'Person detected', text:'A person is visible ahead. Maintain awareness of your surroundings.', detections:[['person','ahead',0.96]], speech:'Person detected ahead. Please remain aware of your surroundings.'},
    {title:'Obstacle detected', text:'A chair-like object is detected on the left side of the image.', detections:[['chair','left',0.89],['backpack','right',0.84]], speech:'Obstacle detected on the left. A backpack is visible on the right.'},
    {title:'Vehicle detected', text:'A vehicle is detected toward the right side of the image.', detections:[['car','right',0.94],['person','ahead',0.91]], speech:'Vehicle detected on the right. Please check your surroundings.'},
    {title:'Pathway visually clear', text:'No priority objects are visible in this simulated sample frame.', detections:[], speech:'The sample scene appears clear of recognised objects.'}
  ];
  const supported = 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window;
  let utteranceId = 0;

  function log(message, level='INFO', source='dashboard') {
    const box=$('activityList'); const empty=box.querySelector('.empty-state'); if(empty) empty.remove();
    const row=document.createElement('div'); row.className='activity-item '+(level==='ERROR'?'error':level==='GUIDANCE'?'guidance':'');
    const time=new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});
    for(const value of [time,level,message]) { const span=document.createElement('span'); span.textContent=value; row.appendChild(span); }
    box.prepend(row); while(box.children.length>35) box.lastElementChild.remove();
  }
  function setMode(mode, description) { state.mode=mode; $('modeLabel').textContent=mode.toUpperCase(); $('modeDescription').textContent=description; $('systemState').textContent=mode==='demo'?'DEMO SIMULATION':mode==='live'?'LIVE VISION CONNECTED':'SYSTEM STANDBY'; }
  function setVoiceStatus(text, sub) { $('voiceStatus').textContent=text; $('voiceSub').textContent=sub; }
  function renderDetections(items, simulated=false) {
    const list=$('detectionList'); list.replaceChildren(); $('detectionCount').textContent=String(items.length).padStart(2,'0'); $('objectCount').textContent=`${items.length} object${items.length===1?'':'s'} in current scene`;
    if(!items.length) { const empty=document.createElement('div'); empty.className='empty-state'; empty.textContent=simulated?'No priority objects in this simulated scene.':'No objects detected in current frame.'; list.appendChild(empty); return; }
    items.forEach(item=>{const chip=document.createElement('div');chip.className='detection-chip';const name=document.createElement('b');name.textContent=item.name;const small=document.createElement('small');small.textContent=`${item.position||'unknown'} · ${Math.round((Number(item.confidence)||0)*100)}% confidence${simulated?' · SIMULATED':''}`;chip.append(name,small);list.appendChild(chip);});
  }
  function setScene(scene, simulated) {
    $('stageTitle').textContent=scene.title; $('stageText').textContent=scene.text; $('currentGuidance').textContent=scene.speech;
    $('simTag').classList.toggle('hidden',!simulated); $('feedBadge').textContent=simulated?'SIMULATED':'LIVE DATA'; $('feedSource').textContent=simulated?'DEMO SCENARIO · SIMULATED':'WINDOWS AGENT · LIVE';
    renderDetections(scene.detections.map(x=>({name:x[0],position:x[1],confidence:x[2]})),simulated);
  }
  function speak(message, key) {
    if(!supported) { $('voiceError').textContent='Speech synthesis is not supported in this browser. Try an updated Chrome or Edge browser.'; setVoiceStatus('Unsupported','Use a supported desktop or mobile browser'); return; }
    if(!state.voiceEnabled || state.muted) { log('Announcement shown; voice is not enabled.','WARNING'); return; }
    const unique=key||message; if(state.lastSpoken.has(unique)) return; state.lastSpoken.add(unique); if(state.lastSpoken.size>150) state.lastSpoken.delete(state.lastSpoken.values().next().value);
    try {
      const utterance=new SpeechSynthesisUtterance(message); utterance.lang='en-US'; utterance.rate=0.96; utterance.pitch=1; utterance.volume=1;
      const id=++utteranceId; state.speaking=false; $('speechPulse').className='pulse active'; $('speechState').textContent='Sending announcement to browser audio…';
      utterance.onstart=()=>{if(id===utteranceId){state.speaking=true;$('speechPulse').className='pulse speaking';$('speechState').textContent='Speaking now';log(`Browser began speaking: ${message}`,'GUIDANCE','browser-voice');}};
      utterance.onend=()=>{if(id===utteranceId){state.speaking=false;$('speechPulse').className='pulse active';$('speechState').textContent='Speech finished';}};
      utterance.onerror=(event)=>{if(id===utteranceId){state.speaking=false;$('speechPulse').className='pulse';$('speechState').textContent='Speech failed';$('voiceError').textContent=`Browser speech error: ${event.error||'unknown error'}. Try enabling voice again.`;setVoiceStatus('Speech error','The browser could not play this announcement');log(`Speech failed: ${event.error||'unknown error'}`,'ERROR');}};
      window.speechSynthesis.speak(utterance);
    } catch(error) { state.speaking=false; $('voiceError').textContent=`Could not start browser speech: ${error.message}`; $('speechState').textContent='Speech failed'; log(`Speech exception: ${error.message}`,'ERROR'); }
  }
  function enableVoice() {
    if(!supported) { $('voiceError').textContent='This browser does not support speech synthesis. Please use current Chrome or Edge.';return; }
    try { const test=new SpeechSynthesisUtterance(''); test.volume=0; window.speechSynthesis.speak(test); state.voiceEnabled=true;state.muted=false;$('muteVoiceBtn').disabled=false;$('muteVoiceBtn').textContent='Ⅱ MUTE';$('voiceError').textContent='';setVoiceStatus('Enabled','Browser speech synthesis is ready');$('speechSupport').textContent='SUPPORTED';$('speechPulse').className='pulse active';$('speechState').textContent='Voice enabled on this device';log('Browser voice enabled by user interaction.','VOICE');speak('AKSHU five point zero browser voice is enabled.','voice-enabled'); }
    catch(error){$('voiceError').textContent=`Voice could not be enabled: ${error.message}`;setVoiceStatus('Failed','Check browser audio permissions');}
  }
  function startDemo() {
    stopDemo(false); state.lastSpoken.clear(); state.demoIndex=0; setMode('demo','DEMO MODE · all detections are simulated, not from a camera');
    $('cameraStatus').textContent='Simulated scene';$('cameraSub').textContent='No camera or local agent used';$('modelStatus').textContent='Demo scenarios';$('agentStatus').textContent='Not required';$('agentSub').textContent='Demo runs independently';$('stopDemoBtn').disabled=false;$('feedBadge').textContent='SIMULATED';
    $('visionStage').classList.add('simulated-stage');$('liveFrame').classList.add('hidden'); $('stageTitle').textContent='Demo sequence starting';$('stageText').textContent='A rotating set of example situations will appear here.';
    log('Demo Mode started. Simulated detections only.','INFO');
    const tick=()=>{const index=state.demoIndex++%demoScenes.length;const scene=demoScenes[index];setScene(scene,true);speak(scene.speech,`demo-${index}-${Math.floor(state.demoIndex/demoScenes.length)}`);log(`[SIMULATED] ${scene.title}`,'GUIDANCE','demo-engine');};
    tick();state.demoTimer=window.setInterval(tick,6500);
  }
  function stopDemo(addLog=true) {if(state.demoTimer){clearInterval(state.demoTimer);state.demoTimer=null;}if(state.mode==='demo'){setMode('idle','Demo stopped. Start Demo Mode to explore without a camera.');$('cameraStatus').textContent='Standby';$('cameraSub').textContent='No camera required for demo';$('modelStatus').textContent='Awaiting input';$('agentStatus').textContent='Disconnected';$('agentSub').textContent='Windows laptop connection';$('stopDemoBtn').disabled=true;$('simTag').classList.add('hidden');$('feedBadge').textContent='NO SIGNAL';$('feedSource').textContent='NO ACTIVE FEED';$('stageTitle').textContent='Awaiting visual input';$('stageText').textContent='Start Demo Mode or connect your Windows agent.';renderDetections([]);if(addLog)log('Demo Mode stopped.','INFO');}}
  async function refreshLive() {
    try {const response=await fetch('/api/status',{cache:'no-store'});if(!response.ok)throw new Error(`HTTP ${response.status}`);const data=await response.json();
      if(state.mode==='demo')return;
      if(data.agent_connected){setMode('live','LIVE VISION MODE · detections originate from your Windows laptop');$('cameraStatus').textContent=data.camera_status;$('cameraSub').textContent='Camera is attached to the local agent';$('modelStatus').textContent=data.model_status;$('agentStatus').textContent='Connected';$('agentSub').textContent='Authenticated agent heartbeat received';$('fpsValue').textContent=`${data.fps} FPS`;$('feedBadge').textContent='LIVE';$('feedSource').textContent='WINDOWS AGENT · LIVE';renderDetections(data.detections||[]);$('currentGuidance').textContent=(data.detections||[]).length?`Live detections: ${data.detections.map(d=>`${d.name} ${d.position}`).join(', ')}`:'No recognised objects in current frame.';
        if(data.frame){$('liveFrame').src=`data:image/jpeg;base64,${data.frame}`;$('liveFrame').classList.remove('hidden');$('stageTitle').textContent='Live annotated camera';$('stageText').textContent='Frame processed locally by YOLO on your Windows laptop.';}else{$('liveFrame').classList.add('hidden');$('stageTitle').textContent='Live agent connected';$('stageText').textContent='Live detections are arriving. Enable browser voice to hear announcements.';}
      } else {if(state.mode==='live')setMode('idle','The local agent heartbeat has expired.');$('agentStatus').textContent='Disconnected';$('agentSub').textContent='Start akshu_agent.py on your laptop';$('cameraStatus').textContent='Agent offline';$('modelStatus').textContent='Not connected';$('feedBadge').textContent='NO SIGNAL';}
    } catch(error){$('agentStatus').textContent='Cloud unavailable';$('agentSub').textContent='Retrying dashboard connection';log(`Status request failed: ${error.message}`,'ERROR');}
  }
  async function pollEvents(){try{const response=await fetch(`/api/events?after=${state.lastEventId}`,{cache:'no-store'});const data=await response.json();for(const event of data.events||[]){state.lastEventId=Math.max(state.lastEventId,event.id);log(event.message,event.level,event.source);if(event.level==='GUIDANCE'&&event.source==='live-agent'&&state.mode!=='demo'){ $('currentGuidance').textContent=event.message;speak(event.message,`live-event-${event.id}`);}}}catch(_){}}
  $('demoBtn').addEventListener('click',startDemo);$('stopDemoBtn').addEventListener('click',()=>stopDemo());$('startLiveBtn').addEventListener('click',()=>{stopDemo(false);setMode('idle','Start the Windows agent with your Render URL and shared token.');$('agentStatus').textContent='Waiting';$('agentSub').textContent='See README: Windows live-agent setup';log('Live mode requires the Windows local agent.','INFO');});
  $('enableVoiceBtn').addEventListener('click',enableVoice);$('testVoiceBtn').addEventListener('click',()=>{if(!state.voiceEnabled){$('voiceError').textContent='First click ENABLE VOICE, then click TEST VOICE.';return;}state.lastSpoken.delete('manual-voice-test');speak('This is AKSHU five point zero. Browser voice test is working.','manual-voice-test');});
  $('muteVoiceBtn').addEventListener('click',()=>{state.muted=!state.muted;if(state.muted){window.speechSynthesis.cancel();$('muteVoiceBtn').textContent='▶ UNMUTE';$('speechState').textContent='Voice muted';$('speechPulse').className='pulse';}else{$('muteVoiceBtn').textContent='Ⅱ MUTE';$('speechState').textContent='Voice enabled';$('speechPulse').className='pulse active';}log(state.muted?'Browser voice muted.':'Browser voice unmuted.','VOICE');});
  $('speechSupport').textContent=supported?'SUPPORTED':'NOT SUPPORTED';if(!supported){setVoiceStatus('Unsupported','Use a current Chrome, Edge, or Safari browser');$('enableVoiceBtn').disabled=true;$('testVoiceBtn').disabled=true;} else {setVoiceStatus('Not enabled','Click ENABLE VOICE to permit browser audio');}
  refreshLive();pollEvents();state.pollTimer=window.setInterval(()=>{refreshLive();pollEvents();},1800);
  window.addEventListener('beforeunload',()=>{if(state.demoTimer)clearInterval(state.demoTimer);if(window.speechSynthesis)window.speechSynthesis.cancel();});
})();
