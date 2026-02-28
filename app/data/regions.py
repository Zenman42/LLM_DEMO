"""Static region reference data for Yandex and Google search engines.

These codes are used by the JustMagic collector:
- Yandex: payload["region"] = str(region_yandex)  (integer code)
- Google: payload["google_lr"] = region_google
  Format: "City,Region,Country\\tGeoTargetID" (Google Ads Criteria ID)
  Example: "Moscow,Moscow,Russia\\t1011969"
"""

YANDEX_REGIONS: list[dict] = [
    # Россия — города
    {"code": 225, "name": "Россия (вся)", "country": "Россия"},
    {"code": 213, "name": "Москва", "country": "Россия"},
    {"code": 2, "name": "Санкт-Петербург", "country": "Россия"},
    {"code": 54, "name": "Екатеринбург", "country": "Россия"},
    {"code": 43, "name": "Казань", "country": "Россия"},
    {"code": 65, "name": "Новосибирск", "country": "Россия"},
    {"code": 66, "name": "Нижний Новгород", "country": "Россия"},
    {"code": 56, "name": "Челябинск", "country": "Россия"},
    {"code": 172, "name": "Уфа", "country": "Россия"},
    {"code": 36, "name": "Ростов-на-Дону", "country": "Россия"},
    {"code": 51, "name": "Самара", "country": "Россия"},
    {"code": 67, "name": "Омск", "country": "Россия"},
    {"code": 47, "name": "Красноярск", "country": "Россия"},
    {"code": 50, "name": "Пермь", "country": "Россия"},
    {"code": 62, "name": "Воронеж", "country": "Россия"},
    {"code": 44, "name": "Волгоград", "country": "Россия"},
    {"code": 38, "name": "Краснодар", "country": "Россия"},
    {"code": 35, "name": "Саратов", "country": "Россия"},
    {"code": 24, "name": "Тюмень", "country": "Россия"},
    {"code": 11119, "name": "Тольятти", "country": "Россия"},
    {"code": 37, "name": "Ижевск", "country": "Россия"},
    {"code": 48, "name": "Барнаул", "country": "Россия"},
    {"code": 68, "name": "Иркутск", "country": "Россия"},
    {"code": 75, "name": "Владивосток", "country": "Россия"},
    {"code": 76, "name": "Хабаровск", "country": "Россия"},
    {"code": 39, "name": "Ярославль", "country": "Россия"},
    {"code": 33, "name": "Калининград", "country": "Россия"},
    {"code": 42, "name": "Оренбург", "country": "Россия"},
    {"code": 53, "name": "Рязань", "country": "Россия"},
    {"code": 55, "name": "Пенза", "country": "Россия"},
    {"code": 57, "name": "Тула", "country": "Россия"},
    {"code": 58, "name": "Липецк", "country": "Россия"},
    {"code": 964, "name": "Сочи", "country": "Россия"},
    {"code": 2, "name": "Санкт-Петербург", "country": "Россия"},
    # Россия — области/регионы
    {"code": 1, "name": "Московская область", "country": "Россия"},
    # Украина
    {"code": 187, "name": "Украина (вся)", "country": "Украина"},
    {"code": 143, "name": "Киев", "country": "Украина"},
    {"code": 144, "name": "Харьков", "country": "Украина"},
    {"code": 145, "name": "Одесса", "country": "Украина"},
    {"code": 141, "name": "Днепр", "country": "Украина"},
    # Беларусь
    {"code": 149, "name": "Беларусь (вся)", "country": "Беларусь"},
    {"code": 157, "name": "Минск", "country": "Беларусь"},
    # Казахстан
    {"code": 159, "name": "Казахстан (весь)", "country": "Казахстан"},
    {"code": 163, "name": "Алматы", "country": "Казахстан"},
    {"code": 164, "name": "Нур-Султан (Астана)", "country": "Казахстан"},
    # Узбекистан
    {"code": 171, "name": "Узбекистан (весь)", "country": "Узбекистан"},
    {"code": 10335, "name": "Ташкент", "country": "Узбекистан"},
]

GOOGLE_REGIONS: list[dict] = [
    # Россия
    {"code": "Russia\t2643", "name": "Россия (вся)", "country": "Russia"},
    {"code": "Moscow,Moscow,Russia\t1011969", "name": "Москва", "country": "Russia"},
    {"code": "Saint Petersburg,Saint Petersburg,Russia\t1006886", "name": "Санкт-Петербург", "country": "Russia"},
    {"code": "Yekaterinburg,Sverdlovsk Oblast,Russia\t1011906", "name": "Екатеринбург", "country": "Russia"},
    {"code": "Kazan,Tatarstan,Russia\t1011940", "name": "Казань", "country": "Russia"},
    {"code": "Novosibirsk,Novosibirsk Oblast,Russia\t1011891", "name": "Новосибирск", "country": "Russia"},
    {"code": "Nizhny Novgorod,Nizhny Novgorod Oblast,Russia\t1011888", "name": "Нижний Новгород", "country": "Russia"},
    {"code": "Chelyabinsk,Chelyabinsk Oblast,Russia\t1011916", "name": "Челябинск", "country": "Russia"},
    {"code": "Ufa,Republic of Bashkortostan,Russia\t1011923", "name": "Уфа", "country": "Russia"},
    {"code": "Rostov-on-Don,Rostov Oblast,Russia\t1011897", "name": "Ростов-на-Дону", "country": "Russia"},
    {"code": "Samara,Samara Oblast,Russia\t1011899", "name": "Самара", "country": "Russia"},
    {"code": "Omsk,Omsk Oblast,Russia\t1011893", "name": "Омск", "country": "Russia"},
    {"code": "Krasnoyarsk,Krasnoyarsk Krai,Russia\t1011879", "name": "Красноярск", "country": "Russia"},
    {"code": "Perm,Perm Krai,Russia\t1011895", "name": "Пермь", "country": "Russia"},
    {"code": "Voronezh,Voronezh Oblast,Russia\t1011913", "name": "Воронеж", "country": "Russia"},
    {"code": "Volgograd,Volgograd Oblast,Russia\t1011911", "name": "Волгоград", "country": "Russia"},
    {"code": "Krasnodar,Krasnodar Krai,Russia\t1011877", "name": "Краснодар", "country": "Russia"},
    {"code": "Saratov,Saratov Oblast,Russia\t1011901", "name": "Саратов", "country": "Russia"},
    {"code": "Tyumen,Tyumen Oblast,Russia\t1011945", "name": "Тюмень", "country": "Russia"},
    {"code": "Tolyatti,Samara Oblast,Russia\t1011900", "name": "Тольятти", "country": "Russia"},
    {"code": "Izhevsk,Udmurt Republic,Russia\t1011947", "name": "Ижевск", "country": "Russia"},
    {"code": "Barnaul,Altai Krai,Russia\t1011855", "name": "Барнаул", "country": "Russia"},
    {"code": "Irkutsk,Irkutsk Oblast,Russia\t1011866", "name": "Иркутск", "country": "Russia"},
    {"code": "Vladivostok,Primorsky Krai,Russia\t1011896", "name": "Владивосток", "country": "Russia"},
    {"code": "Khabarovsk,Khabarovsk Krai,Russia\t1011951", "name": "Хабаровск", "country": "Russia"},
    {"code": "Yaroslavl,Yaroslavl Oblast,Russia\t1011955", "name": "Ярославль", "country": "Russia"},
    {"code": "Kaliningrad,Kaliningrad Oblast,Russia\t1011868", "name": "Калининград", "country": "Russia"},
    {"code": "Sochi,Krasnodar Krai,Russia\t1011878", "name": "Сочи", "country": "Russia"},
    # Украина
    {"code": "Ukraine\t2804", "name": "Украина (вся)", "country": "Ukraine"},
    {"code": "Kiev,Kyiv city,Ukraine\t1012839", "name": "Киев", "country": "Ukraine"},
    {"code": "Kharkiv,Kharkiv Oblast,Ukraine\t1012833", "name": "Харьков", "country": "Ukraine"},
    {"code": "Odessa,Odessa Oblast,Ukraine\t1012837", "name": "Одесса", "country": "Ukraine"},
    {"code": "Dnipro,Dnipropetrovsk Oblast,Ukraine\t1012828", "name": "Днепр", "country": "Ukraine"},
    # Беларусь
    {"code": "Belarus\t2112", "name": "Беларусь (вся)", "country": "Belarus"},
    {"code": "Minsk,Minsk Region,Belarus\t1001304", "name": "Минск", "country": "Belarus"},
    # Казахстан
    {"code": "Kazakhstan\t2398", "name": "Казахстан (весь)", "country": "Kazakhstan"},
    {"code": "Almaty,Almaty Province,Kazakhstan\t1001526", "name": "Алматы", "country": "Kazakhstan"},
    {"code": "Astana,Akmola Province,Kazakhstan\t1001519", "name": "Нур-Султан (Астана)", "country": "Kazakhstan"},
    # Узбекистан
    {"code": "Uzbekistan\t2860", "name": "Узбекистан (весь)", "country": "Uzbekistan"},
    {"code": "Tashkent,Uzbekistan\t1012870", "name": "Ташкент", "country": "Uzbekistan"},
    # Международные
    {"code": "United States\t2840", "name": "США", "country": "United States"},
    {"code": "United Kingdom\t2826", "name": "Великобритания", "country": "United Kingdom"},
    {"code": "Germany\t2276", "name": "Германия", "country": "Germany"},
    {"code": "France\t2250", "name": "Франция", "country": "France"},
    {"code": "Turkey\t2792", "name": "Турция", "country": "Turkey"},
    {"code": "Israel\t2376", "name": "Израиль", "country": "Israel"},
]
