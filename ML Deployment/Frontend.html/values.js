function validateInput() {
  var PPE = document.getElementById("PPE").value;
  var DFA = document.getElementById("DFA").value;
  var RPDE = document.getElementById("RPDE").value;
  var NumPulses = document.getElementById("NumPulses").value;
  var NumPeriodPulses = document.getElementById("NumPeriodPulses").value;
  var result = document.getElementById("result"); 
  if (PPE >= -1 && PPE <= 0 && DFA >= -1 && DFA <= 0 && RPDE >= -1 && RPDE <= 0) {
    result.innerHTML = "Parkinson's Disease Detected";
  } 
  else if (NumPulses > 0 && NumPulses <= 1 && NumPeriodPulses > 0 && NumPeriodPulses <= 1) {
    result.innerHTML = "Parkinson's Disease not Detected";
  }
  else {
    result.innerHTML = "Not enough data to determine Parkinson's Disease status.";
  }
}