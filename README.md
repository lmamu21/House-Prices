# House Prices - Advanced Regression Techniques
* გადავწყვიტე, რომ მონაცემები გამეწმინდა და დამეორგანიზებინა ერთხელ და ის გამომეყენებინა ყველა მოდელზე.

# რეპოზიტორიის სტრუქტურა
* model_experiment.ipynb - მონაცემების წაკითხვა, გაწმენდა, feature-engineering, მოდელის გაწვრთვნა
* model_inference.ipynb - საუკეთესო მოდელი

# Feature Engineering
* პირველ რიგში გადავწყვიტე, რომ გამეწმინდა NA მნიშვნელობები, რათა შემდეგ მარტივად გარდამექმნა რიცხვითი და კატეგორიული სვეტები.
    - PoolQC-ის მნიშვნელობების 99% NA იყო ამიტომ ის უბრალოდ წავშალე
    - MiscFeature ასევე 96% შემთხვევაში იყო NA და მასთან ერთად წავშალე MiscVal რომელის მას აღეწერდა
    - მსგავსად მოვიქეცი Alley-ს შემთხვევაში 93%-იან NA მნიშვნელობებით
    - Fence ცვლადში NA-ს მივანიჭრე 0 მნიშვნელობა ხოლო ყველა დანარჩენს 1 - მივიღე ცვლადი რომელიც აღწერს სახლს ღობე აქვს თუ არა.
    - MasVnrType-ის ნახევარი იყო NA, რაც საკმარისი არ იქნებოდა მთლიანი სვეტის მოსაშორებლად, მაგრამ ManVnrArea აღწერს მის რაოდენობას, ხოლო გადავწყვიტე, რომ ტიპი აღარ იყო საჭირო
    - FirePlaceQu - ში NA მნიშვნელობები შევცვალე 'No Fireplace'-ით და შემდეგ მთლიანად კატეგორიული მნიშვნელობები შევცვალე რაოდენობრივით ordinal მიდგომით
    - Electrical - აქ NA უმნიშვნელოდ იყო ამიტომ ყველაზე ხშირი კატეგორიით შევცვალე
    - LotFrontage - რაოდენობრივი მნიშვნელობის NA-ები შევცვალე 0-ით
    - 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2' - basement-თან დაკავშირებულ ყველა ცვლადში NA ნიშნავდა, რომ სახლს არ ჰქონდა ბეისმენტი, ამიტომ NA შევცვალე 'No Basement'-ით, რაც მოგვიანებით დავამუშავე საბოლოოდ
    - 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond' - მსგავსად მოვიქეცი ავტოფარეხთან დაკავშირებულ ცვლადებზე
    - GarageYrBlt - რიცხვითი NA მნიშვნელობები შევცვალე, ყველა დანარჩენის მედიანით
* შემდეგ კატეგორიული ცვლადები უნდა გადამეყვანა რიცხვითში
    - 'Neighborhood', 'Exterior1st', 'Exterior2nd' - target_encoding მიდგომით შევცვალე
    - Condition1 - აქ იყო 1 ხშირად გამეორებული მნიშვნელობა და დანარაჩენი წვრილად დანაწევრებული რამდენიმე მნიშვნელობა, ამიტომ ვცადე იშვიათი მნიშვნელობები გამეერთიანებინა Others-ში და შემდეგ გამომეყენებინა one-hot_encoding დარჩენილ მნიშვნელობებზე
    - მსგავსად მოვიქეცი SaleType ცვლადზე იგივე მიზეზებით
    - 
    - 'LandContour', 'Utilities', 'LandSlope', 'BsmtFinType', 'Functional', 'GarageFinish', 'BsmtExposure', 'PavedDrive', 'LotShape', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'KitchenQual', 'HeatingQC', 'GarageQual', 'GarageCond'
    ბევრ ცვლადზე შესაძლებელი იყო კატეგორიების პირდაპირ რიცხვითში გადაყვანა, ordinal მიდგომით (ცუდი, საშუალო, კარგი -> 0, 1, 2) გადავიყვანე ეს ცვლადები
    - CentralAir - მხოლოდ yes, no მნიშვნელობების გამო გადავიყვანე ორობით ცვლადში
    - streets, condition_2, RoofMatl, Heating - ამ ცვლადებში ერთი მნიშვნელობის დომინაციის გამო მნიშვნელობები მოვაშორე
    - ამის შემდეგ დარჩა კატეგორიული ცვლადები შედარებით ნაკლები შესაძლო მნიშვნელობით და მათზე one_hot_encoding გადავატარე.

    * Feature Selection
        - Feature Selection-ისთვის გამოვიყენე correlation filter, რასაც დავუწესე 0.8-იანი ზღვარი და ამით 11 სვეტის ამოკლება შევძელი
        - ასევე გამოვიყენე RFE თითოეულ მოდელზე სხვადასხვა მაქსიმალური სვეტით.
    
    * Training
        - ტრეინინგში გავტესტე რამდენიმე კომბინაცია -> LinearRegression, KFoldCrossValidation, StandardScaler, MinMaxScaler და მათი სხვადასხვა ჰიპერპარამეტრები
        - საუკეთესო შედეგი მომცა KFold-მა 10 ნაწილით StandardScaler-თან და RFE(n_features_to_select=20)-თან ერთად.

    * MLFlow Tracking
        - https://dagshub.com/lmamu21/House-Prices.mlflow/#/experiments/6?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D
        - საუკეთესო მოდელი ცალკე model_inference.ipynb-ში ვერ გამოვიყენე.
         