⏺ Here's how to use each tab:

  Tab 1 — Pattern Space: This is your overview. The UMAP scatter shows all 1905 patches as points. Use the "Colour by" dropdown to
  switch between cluster_id (see the 8 discovered pattern types), altitude_group (see if patterns correlate with drone altitude), or
  source_type. Hover over points to see their IDs. Clusters that are tight and well-separated in this view are strong groupings.

  Tab 2 — Cluster Browser: This is where you inspect individual clusters. Pick one from the dropdown and you'll see the actual patch
  images that belong to it — this is how you figure out what each cluster "means" ecologically. The description panel shows the
  auto-generated texture characterization, and the bar chart shows which texture features most distinguish the clusters overall.

  Tab 3 — Temporal: Shows how clusters distribute across temporal phases (pre/during/post rain). Limited for this dataset since there's
   no pre-rain imagery, but if there were phase variation you'd see which pattern types shift over time.

  Tab 4 — Upload & Classify: Drag in a new desert image. The pipeline runs it through DINOv3 → curation projection → UMAP → HDBSCAN and
   tells you which cluster it belongs to, with a confidence score. It appears as a red star on the UMAP scatter, and you see its 10
  nearest neighbor patches from the dataset. This is how you'd classify new field photos.

  Tab 5 — Correlation Workbench: This is for the microbial ecology question. Upload a CSV with lat, lon, and measurement columns (e.g.
  microbial activity, ATP counts). It matches each measurement point to the nearest image patch by GPS, then reports Spearman
  correlations and Random Forest R² between the pattern space coordinates and your measurements. The overlay plot shows measurement
  values on the UMAP scatter — if you see a gradient across the pattern space, that suggests pattern type predicts microbial activity.

  Typical workflow: Start in Tab 2 to understand what each cluster looks like visually. Then Tab 1 to see the big picture. Then Tab 5
  when you have measurement data to test hypotheses.
