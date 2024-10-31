const myForm = document.querySelector('#my-form');
const userInput = document.querySelector('#userId');
const convtypeInput = document.querySelector('#convtype');

myForm.addEventListener('submit', onSubmit);

// load userId 
let userId = localStorage.getItem('userId'); // set userID if exists 
if(userId != '') {
    userInput.value = userId;
}

let conversationType = localStorage.getItem('convType'); // set conversationType if exists 
if(conversationType != '') {
    convtypeInput.value = conversationType;
}
else {
    convtypeInput.value = "normal"  // general conversation
}

console.log(userInput.value);
console.log(convtypeInput.value);

const multiRegionInput = document.querySelector('#multiRegion');
let multi_region = localStorage.getItem('multiRegion'); // set conversationType if exists 
if(multi_region != '') {
    multiRegionInput.value = multi_region;
}
else {
    multiRegionInput.value = "disable"  // general conversation
}
console.log(multiRegionInput.value);

const gradeInput = document.querySelector('#gradeMode');
let grade_mode = localStorage.getItem('gradeMode');
if(grade_mode != '') {
    gradeInput.value = grade_mode;
}
else {
    gradeInput.value = "LLM"  
}
console.log('gradeInput: ', gradeInput.value);

// provisioning
getProvisioningInfo(userId);

function onSubmit(e) {
    e.preventDefault();
    console.log(userInput.value);
    console.log(convtypeInput.value);
    console.log(multiRegionInput.value);

    localStorage.setItem('userId',userInput.value);
    console.log('Save Profile> userId:', userInput.value)    

    localStorage.setItem('convType',convtypeInput.value);
    console.log('Save Profile> convtype:', convtypeInput.value)

    localStorage.setItem('multiRegion',multiRegionInput.value);
    console.log('Save config> multiRegion:', multiRegionInput.value)

    localStorage.setItem('gradeMode',gradeInput.value);
    console.log('Save config> gradeInput:', gradeInput.value)

    window.location.href = "chat.html";
}

function getProvisioningInfo(userId) {
    const uri = "provisioning";
    const xhr = new XMLHttpRequest();

    xhr.open("POST", uri, true);
    xhr.onreadystatechange = () => {
        if (xhr.readyState === 4 && xhr.status === 200) {
            let response = JSON.parse(xhr.responseText);
            let provisioning_info = JSON.parse(response['info']);
            console.log("provisioning info: " + JSON.stringify(provisioning_info));
                        
            let wss_url = provisioning_info.wss_url;
            console.log("wss_url: ", wss_url);

            localStorage.setItem('wss_url',wss_url);
        }
    };

    var requestObj = {
        "userId": userId
    }
    console.log("request: " + JSON.stringify(requestObj));

    var blob = new Blob([JSON.stringify(requestObj)], {type: 'application/json'});

    xhr.send(blob);   
}
