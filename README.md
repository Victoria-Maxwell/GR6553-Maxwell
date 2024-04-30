# GR6553-Maxwell
Final Project for Comp Methods: plotting data for Tropical Storm Cristobal.

All GFS data needs to be requested from NCEI's data archive for the month, day, year and forecast hour.
File can be requested at this link here

https://www.ncei.noaa.gov/has/HAS.FileAppRouter?datasetname=GFSGRB24&subqueryby=STATION&applname=&outdest=FILE
Once the order has been processed and emailed to you, download with the web link (first link in the email), then you can pull the indivdual forecast hour you need.

The CSV file was created from the Flight Level Data from the link here https://www.aoml.noaa.gov/2020-hurricane-field-program-data/#cristobal
The original text file was edited to get rid of header information. It was then edited in Microsoft Excel to become a space deliminated file, every column but windspeed, pressure (need to make sure that it is only going to the lowest pressure of the storm at the time) and time was deleted.
