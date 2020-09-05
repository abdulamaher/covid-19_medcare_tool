const containerDiv = document.getElementById('vizContainer');
const url = "https://public.tableau.com/shared/25WYRRT55";
const options = {
    hideTabs: true,
    height:800,
    width:1200,
};

function initViz(){
    let viz = new tableau.Viz(containerDiv, url, options);
}

document.addEventListener('DOMContentLoaded', initViz);
