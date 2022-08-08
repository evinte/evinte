const box = document.getElementsByClassName('col-25');
function collapse() {
  const elms = document.getElementsByClassName("explanation-text");
  Array.from(elms).forEach((x) => {
    if (x.style.display === "none") {
      x.style.display = "block";
    } else {
      x.style.display = "none";
    }
  })
}