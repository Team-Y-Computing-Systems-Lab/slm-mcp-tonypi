// Loop over input items and add a new field called 'myNewField' to the JSON of each one
const actions = $input.first().json.output;
JSON.parse(actions)["action"].map((act, indx) => {
  return {
    action: act, 
  }
});
return [{
  "run": true,  
}];