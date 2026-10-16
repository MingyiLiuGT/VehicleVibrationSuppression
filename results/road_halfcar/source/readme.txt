This data subset is part of a the LiRA-CD collection
- An Open-Source Dataset for Road Condition Modelling and Research

The subset contains selected standard road condition data from the LiRA project.

The following roads are included in the data subset:
- Highway M3: 51 km (heavy/right lane in each direction tested)
- Highway M14: 30.6 km (heavy/right lane in each direction tested)
- Urban road CPH1 (ring road O2): 50.8 km (heavy/right lane in each driection tested)
- Urban road CPH6: 43.4 km (heavy/right lane in each direction tested)

The subset contains standard road condition measurements from vehicles 
operated by the Danish Road Directorate (DRD). The surveying included: (i) P79 
Profilometer – a van equipped with a beam hosting 25 point lasers that measure 
longitudinal and transverse profiles; (ii) ARAN9000 – a multi-functional road 
scanning vehicle that quantifies road defects and distresses using cameras and 
a Laser Cracking Measurement System (LCMS); and (iii) VIAFRIK – a skid resistance 
device complying with CEN/TS 15901-5 standard.

The standard road condition data (.csv) is stored in a flat format; 
each column represents a road condition parameter, GPS information or timestamp, 
while each row represents a new record in time or space. 

The different data fields from the reference data, i.e., the P79 , VIAFRIK and 
ARAN vehicles (respectively), are described below. Moreover, overview of the data 
collection and filenames all for standard road conditions is given in the end of 
this document.

===============================================================
Data fields in road elevation data from the P79 vehicle:
===============================================================

Column header	Description
Distance [m]	Vehicle distance travelled
Laser 1 [mm]	Elevation Laser 1 (transverse position: -1675 mm)
Laser 2 [mm]	Elevation Laser 2 (transverse position: -1475 mm)
Laser 3 [mm]	Elevation Laser 3 (transverse position: -1300 mm)
Laser 4 [mm]	Elevation Laser 4 (transverse position: -1125 mm)
Laser 5 [mm]	Elevation left wheelpath (transverse position: -1025 mm)
Laser 6 [mm]	Elevation Laser 6 (transverse position: -925 mm)
Laser 7 [mm]	Elevation Laser 7 (transverse position: -825 mm)
Laser 8 [mm]	Elevation Laser 8 (transverse position: -725 mm)
Laser 9 [mm]	Elevation Laser 9 (transverse position: -625 mm)
Laser 10 [mm]	Elevation Laser 10 (transverse position: -525 mm)
Laser 11 [mm]	Elevation Laser 11 (transverse position: -425 mm)
Laser 12 [mm]	Elevation Laser 12 (transverse position: -275 mm)
Laser 13 [mm]	Elevation Laser 13 (transverse position: -75 mm)
Laser 14 [mm]	Elevation Laser 14 (transverse position: 125 mm)
Laser 15 [mm]	Elevation Laser 15 (transverse position: 275 mm)
Laser 16 [mm]	Elevation Laser 16 (transverse position: 375 mm
Laser 17 [mm]	Elevation Laser 17 (transverse position: 475 mm)
Laser 18 [mm]	Elevation Laser 18 (transverse position: 575 mm)
Laser 19 [mm]	Elevation Laser 19 (transverse position: 675 mm)
Laser 20 [mm]	Elevation Laser 20 (transverse position: 775 mm)
Laser 21 [mm]	Elevation right wheelpath (transverse position: 875 mm)
Laser 22 [mm]	Elevation Laser 22 (transverse position: 975 mm)
Laser 23 [mm]	Elevation Laser 23 (transverse position: 1150 mm)
Laser 24 [mm]	Elevation Laser 24 (transverse position: 1325 mm)
Laser 25 [mm]	Elevation Laser 25 (transverse position: 1525 mm)
Lat	Latitude
Lon	Longitude
Højde	Elevation
GeoHøjde	Reference height
Alt	Altitude (‘Højde’ + ‘GeoHøjde’)
Bearing	Calculated vehicle heading (bearing angle)

===============================================================
Data fields of International Roughness Index (IRI), Mean Profile Depth 
(MPD) and wheelpath rut depth from the P79 vehicle:
===============================================================
Column header	Description
Distance [m]	Vehicle distance travelled
IRI (5) [m/km]	Left wheelpath International Roughness Index (IRI)
IRI (21) [m/km]	Right wheelpath International Roughness Index (IRI)
MPD 1	Left wheelpath Mean Profile Depth (MPD)
MPD 2	Right wheelpath Mean Profile Depth (MPD)
Venstre sporkøring [mm]	Left wheelpath rut depth
Højre sporkøring [mm]	Right wheelpath rut depth

===============================================================
Data fields of friction data from the VIAFRIK vehicle:
===============================================================
Column header	Description
Tid [ms]	Time in milliseconds
µ_V [-]	Frictioncoefficient left wheelpath
µ_H [-]	Frictioncoefficient right wheelpath
v_MW_V [km/t]	Speed left wheel
v_MW_H [km/t]	Speed right wheel
Slip_V [%]	Sliprate left wheel
Slip_H [%]	Sliprate right wheel
v_TW [km/t]	Speed towing wheel
TotalDist [m]	Accumulated distance travelled 
Lat 	Latitude
Lon	Longitude
F_vertikal_V [N]	Vertical force on left wheel
F_vertikal_H [N]	Vertical force on right wheel
F_friksjon_V [N]	Frictional force on left wheel
F_friksjon_H [N]	Frictional force on right wheel
Bearing	Calculated vehicle heading (bearing angle)

===============================================================
Data fields of road condition data from the ARAN vehicle:
===============================================================
Column header	Description
L_Route_ID	Unique test identifiction number automatically reported by the ARAN vehicle
DCSTimeStamp	Date and start time of test (number format)
BeginChainage	Chainage at the beginning of measurement window
EndChainage	Chainage at the end of measurement window
Venstre IRI (m/km)	Average Left wheelpath International Roughness Index (IRI)
Højre IRI (m/km)	Average Right wheelpath International Roughness Index (IRI)
Rivninger MeanRI (cm³/m²)	Average Ravelling Index (RI) in cm3/m2
Rivninger MeanExistingRI (cm³/m²)	Average Ravelling Index (RI) in cm3/m2
Rivninger MeanRPI (cm³/m²)	Average Road porosity Index (RPI) in cm3/m2
Rivninger MeanAVC (cm³/m²)	Average Air Void content (AVC) in cm3/m2
Venstre Wheelpath Texture MPD (mm)	Average left wheelpath Mean Profile Depth (MPD) from LCMS lasers
Center Texture MPD (mm)	Average road centre Mean Profile Depth (MPD) from LCMS lasers
Højre Wheelpath Texture MPD (mm)	Average right wheelpath Mean Profile Depth (MPD) from LCMS lasers
Left Texture Lasers MPD (mm)	Average left wheelpath Mean Profile Depth (MPD) from texture lasers in front of the vehicle
Right Texture Lasers MPD (mm)	Average right wheelpath Mean Profile Depth (MPD) from texture lasers in front of the vehicle (MPD)
Krakeleringer Små (m²)	Sum of low severity crocodile crakcs (0.5 <  crack density <  0.8)
Krakeleringer Middelstore (m²)	Sum of medium severity crocodille crakcs (0.8 <  crack density <  1.5)
Krakeleringer Store (m²)	Sum of high severity crocodille crakcs (1.5 <  crack density <  100)
Krakeleringer AVG Width (mm)	Average width of crocodile cracks within section
Krakeleringer MIN Width (mm)	Minimum width of crocodile cracks within section
Krakeleringer MAX Width (mm)	Maximum width of crocodile cracks within section
Krakeleringer MIN Depth (mm)	Minimum depth of crocodile cracks within section
Krakeleringer MAX Depth (mm)	Maximum depth of crocodile cracks within section
Revner På Langs Små (m)	Sum of low severity longitudinal cracks (1 mm <  crack <  5 mm) within section
Revner På Langs Middelstore (m)	Sum of medium severity longitudinal cracks (5 mm <  crack <  30 mm) within section
Revner På Langs Store (m)	Sum of high severity longitudinal cracks (30 mm <  crack <  50 mm) within section
Revner På Langs Sealed (m)	Sum of sealed longitudinal cracks within section (detection using smoothness and pixilation differences between the sealed crack and the surrounding area)
Revner På Langs AVG Width (mm)	Average width of longitudinal cracks within section
Revner På Langs MIN Width (mm)	Minimum width of longitudinal cracks within section
Revner På Langs MAX Width (mm)	Maximum width of longitudinal cracks within section
Revner På Langs MIN Depth (mm)	Minimum depth of longitudinal cracks within section
Revner På Langs MAX Depth (mm)	Maximum depth of longitudinal cracks within section
Transverse Low (m)	Sum of low severity transverse cracks  (1 mm <  crack <  5 mm) within section
Transverse Medium (m)	Sum of medium severity transverse cracks (5 mm <  crack <  30 mm) within section
Transverse High (m)	Sum of high severity transverse cracks (30 mm <  crack <  50 mm) within section
Transverse Sealed (m)	Sum of sealed transverse cracks within section (detection using smoothness and pixilation differences between the sealed crack and the surrounding area)
Transverse AVG Width (mm)	Average width of transverse cracks within section
Transverse MIN Width (mm)	Minimum width of transverse cracks within section
Transverse MAX Width (mm)	Maximum width of transverse cracks within section
Transverse MIN Depth (mm)	Minimum depth of transverse cracks within section
Transverse MAX Depth (mm)	Maximum depth of transverse cracks within section
Slaghuller Max Depth Low (mm)	Maximum depth of low severity potholes within section
Slaghuller Max Depth Medium (mm)	Maximum depth medium severity potholes within section
Slaghuller Max Depth High (mm)	Maximum depth high severity potholes within section
Slaghuller Max Depth Delamination (mm)	Maximum depth of delamination within section
Slaghuller Area Affected Low (cm²)	Sum of low severity potholes in cm2 (Average depth between 10 mm and 25 mm)
Slaghuller Area Affected Medium (cm²)	Sum of medium severity potholes in cm2 (Average depth between 25 mm and 50 mm)
Slaghuller Area Affected High (cm²)	Sum of high severity potholes in cm2 (Average depth higher than 50 mm)
Slaghuller Area Affected Delamination (cm²)	Sum of delamination in cm2 (Average depth between 0 mm and 10 mm)
Middelstore Bleeding Left Wheel Path (%)	Percentage of area with light bleeding left wheelpath (1.50 <  bleeding index <  1.75)
Middelstore Bleeding Right Wheel Path (%)	Percentage of area with light bleeding right wheelpath (1.50 <  bleeding index <  1.75)
Store Bleeding Left Wheel Path (%)	Percentage of area with medium bleeding left wheelpath (1.75 <  bleeding index <  2.00)
Store Bleeding Right Wheel Path (%)	Percentage of area with medium bleeding right wheelpath (1.75 <  bleeding index <  2.00)
Meget Store Bleeding Left Wheel Path (%)	Percentage of area with severe bleeding left wheelpath (bleeding index > 2.00)
Meget Store Bleeding Right Wheel Path (%)	Percentage of area with severe bleeding right wheelpath (bleeding index > 2.00)
LRUT Straight Edge (mm)	Average rut depth left wheelpath, calculated using the StraightEdge method 
RRUT Straight Edge (mm)	Average rut depth right wheelpath, calculated using the StraightEdge method 
LRUT Wire (mm)	Average rut depth left wheelpath, calculated using the WIRE method 
RRUT Wire (mm)	Average rut depth right wheelpath, calculated using the WIRE method 
Venstre Exterior Single Stone (cm²)	Sum of area with single stone (loss) left edge
Venstre Exterior Multi Stone (cm²)	Sum of area with multi stone (loss) left edge 
Venstre Wheel Path Single Stone (cm²)	Sum of area with single stone (loss) left wheelpath 
Venstre Wheel Path Multi Stone (cm²)	Sum of area with multi stone (loss) left wheelpath 
Centre Single Stone (cm²)	Sum of area with single stone (loss) central zone 
Centre Multi Stone (cm²)	Sum of area with multi stone (loss) central zone (see road zone definition)
Højre Wheel Path Single Stone (cm²)	Sum of area with single stone (loss) right wheelpath (see road zone definition)
Højre Wheel Path Multi Stone (cm²)	Sum of area with multi stone (loss) right wheelpath 
Højre Exterior Single Stone (cm²)	Sum of area with single stone (loss) right edge (see road zone definition)
Højre Exterior Multi Stone (cm²)	Sum of area with multi stone (loss) right edge (see road zone definition)
Latitude From (deg)	Latitude at the start of the measurement window
Latitude To (deg)	Latitude at the end of the measurement window
Longitude From (deg)	Longitude at the start of the measurement window
Longitude To (deg)	Longitude at the end of the measurement window
Heading (deg)	ARAN vehicle heading (bearing angle)
Elevation (m)	Altitude
Lat	Average latitude of the measurement window
Lon	Average longitude of the measurement window
Alt	Altitude
Heading	ARAN vehicle heading (bearing angle)
Bearing	Calculated vehicle heading (bearing angle)

===============================================================
Overview of the data collection of standard road conditions:
===============================================================
Vehicle	Spatial resolution [m]	Date	Avg. Air Temp. [°]	Surface	Weather	Filename
ARAN9000	1.0	12-08-2020	22	Dry	Sunny	cph1_aran_vh
ARAN9000	1.0	12-08-2020	22	Dry	Sunny	cph1_aran_hh
VIAFRIK	5.0	19-11-2019	6	Dry	Cloudy	cph1_fric_vh
VIAFRIK	5.0	19-11-2019	6	Dry	Cloudy	cph1_fric_hh
P79	10.0	10-09-2020	15	Dry	Sunny	cph1_iri_mpd_rut_vh
P79	10.0	10-09-2020	15	Dry	Sunny	cph1_iri_mpd_rut_hh
P79	0.1	10-09-2020	15	Dry	Sunny	cph1_zp_vh
P79	0.1	10-09-2020	15	Dry	Sunny	cph1_zp_hh
ARAN9000	10.0	30-08-2021	19	Dry	Sunny	cph6_aran_vh
ARAN9000	10.0	30-08-2021	19	Dry	Sunny	cph6_aran_hh
VIAFRIK	5.0	19-11-2019	6	Dry	Cloudy	cph6_fric_vh
VIAFRIK	5.0	19-11-2019	6	Dry	Cloudy	cph6_fric_hh
P79	10.0	04-06-2020	16	Dry	Sunny	cph6_iri_mpd_rut_vh
P79	10.0	04-06-2020	16	Dry	Sunny	cph6_iri_mpd_rut_hh
P79	0.1	04-06-2020	16	Dry	Sunny	cph6_zp_vh
P79	0.1	04-06-2020	16	Dry	Sunny	cph6_zp_hh
ARAN9000	1.0	13-08-2020	23	Dry	Sunny	m3_aran_vh
ARAN9000	1.0	13-08-2020	23	Dry	Sunny	m3_aran_hh
VIAFRIK	5.0	18-11-2019	9	Dry	Cloudy	m3_fric_vh
VIAFRIK	5.0	18-11-2019	9	Dry	Cloudy	m3_fric_hh
P79	10.0	10-09-2020	16	Dry	Sunny	m3_iri_mpd_rut_vh
P79	10.0	10-09-2020	16	Dry	Sunny	m3_iri_mpd_rut_hh
P79	0.1	10-09-2020	16	Dry	Sunny	m3_zp_vh
P79	0.1	10-09-2020	16	Dry	Sunny	m3_zp_hh
ARAN9000	1.0	13-08-2020	23	Dry	Sunny	m13_aran_vh
ARAN9000	1.0	13-08-2020	23	Dry	Sunny	m13_aran_hh
VIAFRIK	5.0	14-08-2020	23	Dry	Sunny	m13_fric_vh
VIAFRIK	5.0	14-08-2020	23	Dry	Sunny	m13_fric_hh
P79	10.0	10-09-2020	16	Dry	Sunny	m13_iri_mpd_rut_vh
P79	10.0	10-09-2020	16	Dry	Sunny	m13_iri_mpd_rut_hh
P79	0.1	10-09-2020	16	Dry	Sunny	m13_zp_vh
P79	0.1	10-09-2020	16	Dry	Sunny	m13_zp_hh

