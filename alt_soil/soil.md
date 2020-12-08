## USDA Soil Report Resources

I generated a soil report from the USDA at their web soil survey (https://websoilsurvey.nrcs.usda.gov/app/). I used "***CO645** - Arapaho-Roosevelt National Forest Area, Parts of Boulder, Clear Creek Gilpin, Grand, Park, and Larimer Counties - USFS*" as the Area of Interest (AOI). I honestly don't know how helpful it's going to be. It's pretty dense. At least it has a halfway **decent Glossary**.

I also found this National Soil Survey Handbook, which might be useful in demystifying some of the soil codes (https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/scientists/?cid=nrcs142p2_054242)

Soil Taxonomy Link (https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/home/?cid=nrcs142p2_053577) These are also pretty long and horrible, but we'll dig in. We see on pp. 119-124 (ch.6) of [the horrible pdf](https://www.nrcs.usda.gov/Internet/FSE_DOCUMENTS/nrcs142p2_051232.pdf) that soils are taxonomically(is that a word?) ordered in the following levels:

> Orders > Suborders > Great Groups > Subgroups > Families > Series 

We are given **families** in the data. 

#### Post EDA Proposal to Refactor Soil

So, looking at the data. Every observation has exactly 1 "soil type". I had anticipated that they would have multiple types in the same observations. However, that's not the case. If we inspect the soil types, we see that they are most accurately described as a "bag" of predictors. So, I think we want to "unpack" them. It seems like for example soils 39 and 40 are pretty similar. We would ideally like to reflect that similarity. Right now, there's none of it.

Actually, I think A resonable ways to do this is as follows, I think we essentially want to split-on/remove ["famly","families","complex"] and split on commas and hyphens. Those seem important maybe if someone wanted to tell exactly what type of soil is in a place, but I don't think it's necessary for our purposes. Then once we've split on those, we can make each of those a separate category. For example, let's look at types 39 and 40 for a demo. 

39. Moran family - Cryorthents - Leighcan family complex, extremely stony.
40. Moran family - Cryorthents - Rock land complex, extremely stony.

I would propose splitting theses into 4 categories each. 5 between the 2 of them. Then we get some overlap between these soils that seem pretty similar but are just completely separate entities right now. The original and transformed values tables would be as follows

|Observation|Soil 39|Soil 40|
|-----------|-------|-------|
|          1|      1|      0|
|          2|      0|      1|

|Observation|Moran|Cryorthents|Leighcan|Rock land|extremely stony|
|-----------|-----|-----------|--------|---------|---------------|
|         1 |   1 |         1 |      1 |       0 |             1 |
|         2 |   1 |         1 |      0 |       1 |             1 |

#### Series Information.

*Maybe Junk. I'm not sure if this is helpful. This is "series" information which is like a degree finer tuned than thte "family" information that they give us. They might be using "family" in place of "series" in the description of the soils. Not sure. I couldn't bear to delete these links etc. In case our interpretation of what they're saying changes later*

~~Okay, I'm closing in. **This looks like the stuff we want!** (https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/survey/class/data/?cid=nrcs142p2_053583). **We want series name**. It looks like the other info on the soil might just be extra information that will potentially obscure it. I'll do some EDA to determine frequency that different classifications show up. BUT, I think that "Leighcan" etc. might be the only imporatnt part and the other stuff might just be fluff.~~

* ~~https://soilseries.sc.egov.usda.gov/OSD_Docs/B/BULLWARK.html~~
* ~~https://soilseries.sc.egov.usda.gov/OSD_Docs/C/CATHEDRAL.html~~
* ~~https://soilseries.sc.egov.usda.gov/OSD_Docs/G/GOTHIC.html~~
* ~~https://soilseries.sc.egov.usda.gov/OSD_Docs/L/LEIGHCAN.html~~
* ~~https://soilseries.sc.egov.usda.gov/OSD_Docs/M/MORAN.html~~
