function initViz(){
    var containerDiv = document.getElementById("treemap")
    url = "https://public.tableau.com/views/cluster_treemap/NeedTreemap?:language=en&:retry=yes&:display_count=y&:origin=viz_share_link"

    var viz = new tableau.Viz(containerDiv, url);
}
