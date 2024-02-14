#!/bin/sh

for graph in TwitterSmall Erdos992 Advogato PagesGovernment WikiElec HepPh Anybeat PagesCompany AstroPh CondMat Gplus Brightkite Themarker Slashdot BlogCatalog WikiTalk Gowalla Academia GooglePlus Citeseer MathSciNet TwitterFollows; do
    julia main_simulation.jl $graph 0.1 10 100 polarized powerlaw
done