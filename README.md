### About the dataset
#### Sites
HARV: Harvard Forest, deciduous boradleaf forest, AmeriFlux code: US-xHA   
https://www.neonscience.org/field-sites/harv   
https://ameriflux.lbl.gov/sites/siteinfo/US-xHA

STEI: Steigerwaldt Land Services, deciduous boradleaf forest, AmeriFlux code: US-xST   
https://www.neonscience.org/field-sites/stei   
https://ameriflux.lbl.gov/sites/siteinfo/US-xST/

#### Columns
TA_F: half-hourly air temperature   
VPD_F: vapor pressure deficit (higher values mean the air is drier)   
P_F: half-hourly precipitation   
GPP: Gross Primary Productivty   
RECO: Ecosystem Respiration   

### Modeling ecosystem respiration
Ecosystem respiration (RECO) is the total release of CO₂ from an ecosystem, combining autotrophic respiration (plants) and heterotrophic respiration (microbes decomposing organic matter). It is controlled by several interacting biophysical and biological factors:

#### Explicitly modeled drivers

- **Temperature**  
  RECO generally increases exponentially with temperature because enzymatic activity in plants and microbes accelerates as conditions warm. This is often the dominant short-term control, particularly in temperate and boreal ecosystems.

- **Soil moisture (proxied by precipitation and VPD)**  
  RECO responds non-linearly to soil moisture availability. Moderate moisture promotes microbial activity and root respiration, while very dry conditions suppress respiration and waterlogged (anoxic) soils limit aerobic decomposition. Precipitation and vapor pressure deficit (VPD) are used as proxies for soil water availability.

- **Vegetation and productivity (proxied by GPP)**  
  Ecosystems with greater biomass, leaf area, and gross primary productivity (GPP) tend to exhibit higher autotrophic respiration, linking RECO closely to photosynthetic activity over seasonal timescales.

#### Other factors not explicitly modeled

- **Substrate availability**  
  The quantity and quality of organic carbon (e.g., litter inputs, root exudates, and soil organic matter) strongly regulate heterotrophic respiration. Fresh carbon inputs and higher productivity typically enhance RECO.

- **Nutrient availability**  
  Nitrogen and other nutrients can stimulate or suppress RECO by altering plant growth, microbial efficiency, and decomposition rates, with effects that depend on ecosystem context.

- **Disturbance and land management**  
  Disturbances such as fire, harvest, drought, grazing, and land-use change modify carbon pools, microclimate, and microbial communities, often leading to abrupt shifts in RECO.

#### A basic physical model
$$
R_{\mathrm{ECO}} = R_{\mathrm{ref}} \times f(T_A) \times f(P)
$$

$$
R_{\mathrm{ref}} = a_0 + a_1 \,\mathrm{SOC} + a_2 \,\mathrm{GPP}
$$

$$
f(T_A) = \exp\!\left[ E_0 \left( \frac{1}{T_{\mathrm{ref}} - T_0} - \frac{1}{T_A - T_0} \right) \right]
$$

$$
f(P) = \frac{\alpha k + P (1 - \alpha)}{k + P (1 - \alpha)}
$$

**Free parameters (to be estimated):**  
\(a_0, a_1, a_2, E_0, \alpha, k\)

Note f(P) basically indicates a function of water availability. f(VPD) could work as well.
SOC is assumed to be a constant for one single site.
