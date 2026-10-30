from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict


# ============================================================
# 1. PAGINA EN INSTELLINGEN
# ============================================================

st.set_page_config(
    page_title="Amsterdam per buurt",
    page_icon="🏘️",
    layout="wide",
)

DATA_MAP = Path(__file__).parent / "data" / "processed"

KENMERKEN = [
    "mediaan_inkomen",
    "bevolkingsdichtheid",
    "percentage_koop",
]

BLAUW = "#2166ac"
ROOD = "#b2182b"


def euro(waarde):
    """Toon een bedrag met Nederlandse duizendtallen."""
    return "€ " + f"{waarde:,.0f}".replace(",", ".")


# ============================================================
# 2. DATA INLEZEN EN SCORES BEREKENEN
# ============================================================

@st.cache_data
def laad_data():
    model5 = pd.read_csv(
        DATA_MAP / "modelresultaten_buurten.csv",
        dtype={"buurtcode_schoon": str},
    )

    model10 = pd.read_csv(
        DATA_MAP / "modelresultaten_min10.csv",
        dtype={"buurtcode_schoon": str},
    )

    buurten = pd.read_csv(
        DATA_MAP / "buurtkenmerken_en_aanbod.csv",
        dtype={"buurtcode_schoon": str},
    )
    
    dekking = pd.read_csv(
        DATA_MAP / "buurtdekking.csv",
        dtype={"cbsCode": str},
    )

    stabiliteit_buurten = pd.read_csv(
        DATA_MAP / "stabiliteit_buurten.csv",
        dtype={"buurtcode_schoon": str},
    ).set_index("buurtcode_schoon")

    with open(
        DATA_MAP / "buurten_model.geojson",
        encoding="utf-8",
    ) as bestand:
        geojson = json.load(bestand)

    

    return model5, model10, buurten, geojson, dekking, stabiliteit_buurten


@st.cache_data
def bereken_scores(data):
    """Beoordeel opgeslagen voorspellingen en dezelfde referentie."""
    y = data["mediaan_prijs_m2"]
    voorspeld = data["voorspeld_m2_cv"]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    referentie = cross_val_predict(
        DummyRegressor(strategy="mean"),
        data[KENMERKEN],
        y,
        cv=cv,
    )

    mae = mean_absolute_error(y, voorspeld)
    mae_referentie = mean_absolute_error(y, referentie)

    return {
        "r2": r2_score(y, voorspeld),
        "mae": mae,
        "mae_referentie": mae_referentie,
        "verbetering": (1 - mae / mae_referentie) * 100,
    }


try:
    model5, model10, alle_buurten, geojson, dekking, stabiliteit_buurten = laad_data()
except FileNotFoundError as fout:
    st.error(
        "Een databestand ontbreekt. Voer eerst de exportcel "
        "in je notebook uit."
    )
    st.code(str(fout))
    st.stop()


# ============================================================
# 3. TITEL EN ALGEMENE FILTER
# ============================================================

st.title("Amsterdam per buurt")

st.markdown(
    "### Welke buurten hebben een hogere vraagprijs per m² "
    "dan hun buurtkenmerken voorspellen?"
)

st.caption(
    "Funda-aanbod oktober 2026 · "
    "Alle buurtkenmerken uit 2024"
)

st.markdown(
    "**Belangrijkste bevinding:** Willemsparkbuurt-Noord, "
    "Zuidas-Noord en Baaibuurt-West hebben bij de hoofdselectie "
    "van minimaal vijf advertenties de grootste positieve "
    "afwijkingen van de modelvoorspelling."
)

st.caption(
    "We vergelijken mediane vraagprijzen uit het aangeboden "
    "woningpakket. Een hoge afwijking is geen bewijs van "
    "overwaardering."
)

with st.sidebar:
    st.header("Selectie")

    minimum = st.radio(
        "Minimaal aantal advertenties per buurt",
        options=[5, 10],
        horizontal=True,
        help=(
            "Bij elke keuze gebruiken we de bijbehorende "
            "opnieuw berekende modelresultaten."
        ),
    )

    st.info(
        "De selectie geldt voor alle tabbladen. "
        "Iedere buurt telt even zwaar in het model."
    )

data = (model5 if minimum == 5 else model10).copy()
scores = bereken_scores(data)

# Maak labels voor de buurtzoeker.
data["keuzelabel"] = (
    data["buurt"] + " — " + data["buurtcode_schoon"]
)

keuzes = data.sort_values("buurt")["keuzelabel"].tolist()

standaard = data.loc[
    data["afwijking_m2_cv"].idxmax(), "keuzelabel"
]

with st.sidebar:
    gekozen_label = st.selectbox(
        "Zoek een buurt",
        options=keuzes,
        index=keuzes.index(standaard),
        key=f"buurtkeuze_{minimum}",
    )

gekozen = data.loc[
    data["keuzelabel"].eq(gekozen_label)
].iloc[0]

# Algemene kengetallen.
kolom1, kolom2, kolom3 = st.columns(3)

kolom1.metric("Onderzochte buurten", len(data))
kolom2.metric(
    "Gemiddelde voorspellingsfout per m²",
    euro(scores["mae"]),
    help=(
        "Het gemiddelde absolute verschil tussen voorspelde "
        "en waargenomen buurtmedianen bij kruisvalidatie. "
        "Dit is geen onzekerheidsmarge per buurt."
    ),
)
kolom3.metric(
    "Verbetering tegenover eenvoudig gemiddelde",
    f"{scores['verbetering']:.1f}%",
    help=(
        "De afname van de gemiddelde absolute voorspellingsfout "
        "ten opzichte van een model dat voor iedere buurt het "
        "gemiddelde van de trainingsbuurten voorspelt."
    ),
)

tab_kaart, tab_model, tab_data = st.tabs([
    "Buurtkaart",
    "Model en betrouwbaarheid",
    "Data en keuzes",
])


# ============================================================
# 4. TABBLAD: BUURTKAART
# ============================================================

with tab_kaart:
    st.subheader("Waar wijkt het aanbod af van de voorspelling?")

    kaartkolom, detailkolom = st.columns([3, 1])

    with kaartkolom:
        # Grijze achtergrond: alle buurten in de opgehaalde kaartlaag.
        achtergrond = pd.DataFrame({
            "cbsCode": [
                feature["properties"]["cbsCode"]
                for feature in geojson["features"]
            ]
        })
        achtergrond = achtergrond.merge(
            dekking,
            on= "cbsCode",
            how= 'left',
            validate='one_to_one'
        )
        
        if minimum == 10:
            codes_5_tot_9 = model5.loc[
                model5["aantal_woningen"].lt(10),
                "buurtcode_schoon",
            ]

            achtergrond.loc[
                achtergrond["cbsCode"].isin(codes_5_tot_9),
                "status",
            ] = "5 tot 9 advertenties; minimum is 10"

        achtergrond["achtergrond"] = "Geen modelresultaat"

        fig = px.choropleth_map(
            achtergrond,
            geojson=geojson,
            locations="cbsCode",
            featureidkey="properties.cbsCode",
            color="achtergrond",
            color_discrete_map={
                "Geen modelresultaat": "#dddddd"
            },
            hover_name="cbsCode",
            hover_data={
                "cbsCode": True,
                "achtergrond": False,
                "status": True,
            },
            labels={"status":"Datadekking"},
            map_style="carto-positron",
            center={"lat": 52.35, "lon": 4.94},
            zoom=10,
        )

        # Gebruik de waarden uit het gekozen model, ook bij minimum 10.
        gekleurde_kaart = px.choropleth_map(
            data,
            geojson=geojson,
            locations="buurtcode_schoon",
            featureidkey="properties.cbsCode",
            color="afwijking_m2_cv",
            color_continuous_scale="RdBu_r",
            range_color=[-3000, 3000],
            hover_name="buurt",
            hover_data={
                "buurtcode_schoon": False,
                "afwijking_m2_cv": ":.0f",
                "mediaan_prijs_m2": ":.0f",
                "voorspeld_m2_cv": ":.0f",
                "aantal_woningen": True,
            },
            labels={
                "afwijking_m2_cv": "Afwijking €/m²",
                "mediaan_prijs_m2": "Mediaan €/m²",
                "voorspeld_m2_cv": "Voorspeld €/m²",
                "aantal_woningen": "Advertenties",
            },
        )

        for trace in gekleurde_kaart.data:
            fig.add_trace(trace)

        fig.update_layout(
            coloraxis=gekleurde_kaart.layout.coloraxis,
            height=560,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
        )

        geselecteerde_buurt = data.loc[
            data["buurtcode_schoon"].eq(
                gekozen["buurtcode_schoon"]
            )
        ]

        highlight = px.choropleth_map(
            geselecteerde_buurt,
            geojson=geojson,
            locations="buurtcode_schoon",
            featureidkey="properties.cbsCode",
            color="afwijking_m2_cv",
            color_continuous_scale="RdBu_r",
            range_color=[-3000, 3000],
            hover_name="buurt",
            hover_data={
                "buurtcode_schoon": False,
                "afwijking_m2_cv": ":.0f",
                "aantal_woningen": True,
            },
            labels={
                "afwijking_m2_cv": "Afwijking €/m²",
                "aantal_woningen": "Advertenties",
            },
        )

        for trace in highlight.data:
            trace.update(
                marker_line_width=4,
                marker_line_color="#FFFF00",
                showscale=False,
            )
            fig.add_trace(trace)

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Rood = boven voorspelling · Blauw = onder voorspelling · "
            "Grijs = geen modelresultaat. "
            "Kleurschaal begrensd op ± €3.000/m²; "
            "de aanwijstekst toont de werkelijke afwijking."
            "Gele omtrek = geselecteerde buurt"
        )

    with detailkolom:
        st.subheader(gekozen["buurt"])

        st.metric(
            "Mediane vraagprijs per m²",
            euro(gekozen["mediaan_prijs_m2"]),
        )
        st.metric(
            "Voorspelde vraagprijs per m²",
            euro(gekozen["voorspeld_m2_cv"]),
        )
        st.metric(
            "Afwijking per m²",
            euro(gekozen["afwijking_m2_cv"]),
        )

        voorspeld = gekozen["voorspeld_m2_cv"]

        if voorspeld > 0:
            afwijking_pct = (
                gekozen["afwijking_m2_cv"] / voorspeld * 100
            )
            st.write(
                f"**{afwijking_pct:+.1f}%** tegenover de voorspelling"
            )

        st.divider()

        st.write(
            f"**Advertenties:** {int(gekozen['aantal_woningen'])}"
        )
        st.write(
            "**Nieuwbouw in het aanbod:** "
            f"{gekozen['aandeel_nieuwbouw_pct']:.1f}%"
        )

    with st.expander(
        f"Hoe stabiel is de voorspelling voor {gekozen['buurt']}?"
    ):
        if minimum == 5:
            buurtcode = gekozen["buurtcode_schoon"]
            stab = stabiliteit_buurten.loc[buurtcode]

            links_stab, rechts_stab = st.columns(2)

            links_stab.metric(
                "Laagste voorspelling",
                euro(stab["voorspelling_min"]) + "/m²",
            )

            rechts_stab.metric(
                "Hoogste voorspelling",
                euro(stab["voorspelling_max"]) + "/m²",
            )

            st.write(
                "De waargenomen vraagprijsmediaan ligt bij "
                f"**{int(stab['keer_boven_voorspelling'])} "
                "van de 20 trainingsverdelingen** boven "
                "de voorspelling."
            )

            st.caption(
                "Dezelfde buurten zijn twintig keer anders "
                "verdeeld over trainings- en testgroepen. "
                "Dit bereik is geen betrouwbaarheidsinterval "
                "en meet geen prijsontwikkeling door de tijd."
            )
        else:
            st.write(
                "Deze controle is uitgevoerd voor het hoofdmodel "
                "met minimaal vijf advertenties. Kies die selectie "
                "om de resultaten te bekijken."
            )

        st.caption(
            "Kies een andere buurt via de zoekfunctie links. "
            "Wijs buurten op de kaart aan voor hun cijfers."
        )

    st.info(
        "Een positieve afwijking betekent dat het aangeboden "
        "woningpakket duurder is dan dit model voorspelt. "
        "Dit bewijst geen overwaardering of afzonderlijke buurtpremie."
    )

    if gekozen['buurtcode_schoon']=="BU0363KC02":
            st.divider()
            st.subheader("Verdieping: Willemsparkbuurt-Noord")
    
            st.write(
                "Deze buurt heeft in de hoofdselectie met minimaal vijf "
                "advertenties de grootste positieve modelafwijking. "
                "Wat hebben we daarover kunnen vaststellen?"
            )
    
            # Gebruik hier altijd het hoofdmodel met minimaal 5 advertenties.
            willemspark = model5.loc[
                model5["buurtcode_schoon"].eq("BU0363KC02")
            ].iloc[0]
    
            kolom_a, kolom_b, kolom_c = st.columns(3)
    
            kolom_a.metric(
                "Waargenomen vraagprijsmediaan",
                euro(willemspark["mediaan_prijs_m2"]) + "/m²",
            )
    
            kolom_b.metric(
                "Voorspelde vraagprijsmediaan",
                euro(willemspark["voorspeld_m2_cv"]) + "/m²",
            )
    
            kolom_c.metric(
                "Boven modelvoorspelling",
                euro(willemspark["afwijking_m2_cv"]) + "/m²",
            )
    
            links_detail, rechts_detail = st.columns(2)
    
            with links_detail:
                st.markdown("**Wat hebben we vastgesteld?**")
    
                st.markdown(
                    "- De buurtmediaan is gebaseerd op **16 advertenties**.\n"
                    "- Bij het weglaten van één advertentie ligt de mediaan "
                    "tussen **€17.990 en €19.172 per m²**. Eén advertentie "
                    "verklaart de hoge mediaan dus niet.\n"
                    "- Het mediane woonoppervlak is **212,5 m²**.\n"
                    "- **6,2%** van het aanbod is nieuwbouw; nieuwbouw "
                    "domineert dit aanbod niet."
                )
    
            with rechts_detail:
                st.markdown("**Wat blijft onverklaard?**")
    
                st.write(
                    "Ons model gebruikt alleen inkomen, bevolkingsdichtheid "
                    "en het percentage koopwoningen. Woningkwaliteit, "
                    "renovatieniveau en specifieke ligging ontbreken. "
                    "Die kunnen bijdragen aan het verschil, maar hun "
                    "bijdrage is niet afzonderlijk onderzocht."
                )
    
                st.write(
                    "Meerdere advertenties liggen bovendien in dezelfde "
                    "gebouwen. De 16 advertenties vertegenwoordigen "
                    "daarom niet noodzakelijk 16 onafhankelijke situaties."
                )
    
            st.info(
                "Conclusie: de positieve modelafwijking blijft bestaan "
                "bij twintig verschillende trainingsverdelingen, en de "
                "hoge vraagprijsmediaan wordt niet door één advertentie "
                "veroorzaakt. De afwijking is een aanleiding voor nader "
                "woningonderzoek, geen bewijs van overwaardering."
            )
    
            st.caption(
                "Deze verdieping gebruikt het hoofdmodel met "
                "minimaal vijf advertenties, ook wanneer links "
                "de ondergrens van tien is geselecteerd."
            )

    st.subheader("Grootste afwijkingen")

    richting = st.radio(
        "Toon buurten",
        ["Boven voorspelling", "Onder voorspelling"],
        horizontal=True,
    )

    positief = richting == "Boven voorspelling"

    ranglijst = data.loc[
        data["afwijking_m2_cv"].gt(0)
        if positief
        else data["afwijking_m2_cv"].lt(0)
    ]

    ranglijst = (
        ranglijst.nlargest(10, "afwijking_m2_cv")
        if positief
        else ranglijst.nsmallest(10, "afwijking_m2_cv")
    )

    ranglijst = ranglijst.sort_values("afwijking_m2_cv")

    fig_rang = px.bar(
        ranglijst,
        x="afwijking_m2_cv",
        y="buurt",
        orientation="h",
        hover_data={"aantal_woningen": True},
        labels={
            "afwijking_m2_cv": "Afwijking (€ per m²)",
            "buurt": "",
            "aantal_woningen": "Advertenties",
        },
    )
    fig_rang.update_traces(
        marker_color=ROOD if positief else BLAUW
    )
    fig_rang.update_layout(
        height=430,
        margin=dict(l=0, r=20, t=10, b=0),
    )
    
    st.plotly_chart(fig_rang, use_container_width=True)
    
   

   


# ============================================================
# 5. TABBLAD: MODEL EN BETROUWBAARHEID
# ============================================================

with tab_model:
    st.subheader("Hoe goed voorspelt het model?")

    st.write(
        "Een lineaire regressie voorspelt de mediane vraagprijs "
        "per m² uit huishoudinkomen, bevolkingsdichtheid en het "
        "percentage koopwoningen."
    )

    st.caption(
        "Vijfvoudige kruisvalidatie: iedere buurt wordt voorspeld "
        "door een model dat die buurt niet als trainingsvoorbeeld "
        "heeft gebruikt. De uitkomsten zijn geen causale effecten."
    )

    links, rechts = st.columns([2, 1])

    with links:
        fig_scatter = px.scatter(
            data,
            x="voorspeld_m2_cv",
            y="mediaan_prijs_m2",
            hover_name="buurt",
            hover_data={"aantal_woningen": True},
            labels={
                "voorspeld_m2_cv": "Voorspelde mediaan (€ per m²)",
                "mediaan_prijs_m2": "Waargenomen vraagprijsmediaan (€ per m²)",
                "aantal_woningen": "Advertenties",
            },
        )

        onder = min(
            data["voorspeld_m2_cv"].min(),
            data["mediaan_prijs_m2"].min(),
        )
        boven = max(
            data["voorspeld_m2_cv"].max(),
            data["mediaan_prijs_m2"].max(),
        )

        fig_scatter.add_shape(
            type="line",
            x0=onder,
            y0=onder,
            x1=boven,
            y1=boven,
            line=dict(color="grey", dash="dash"),
        )

        # Markeer de gekozen buurt.
        fig_scatter.add_scatter(
            x=[gekozen["voorspeld_m2_cv"]],
            y=[gekozen["mediaan_prijs_m2"]],
            mode="markers",
            marker=dict(
                size=15,
                color=ROOD,
                symbol="diamond",
            ),
            name=gekozen["buurt"],
            hovertext=[gekozen["buurt"]],
        )

        st.plotly_chart(
            fig_scatter,
            use_container_width=True,
        )

        st.caption(
            "Op de stippellijn zijn de voorspelde en waargenomen "
            "vraagprijsmediaan gelijk. Iedere stip is één buurt. "
            "Boven de lijn ligt de vraagprijsmediaan hoger dan voorspeld; "
            "eronder lager."
        )

    with rechts:
        st.metric("R² bij kruisvalidatie", f"{scores['r2']:.3f}")

        st.write(
            "Het model laat een groot deel van de verschillen "
            "tussen buurten onverklaard."
        )

        foutvergelijking = pd.DataFrame({
            "Model": ["Eenvoudig gemiddelde", "Buurtkenmerken"],
            "Gemiddelde fout": [
                scores["mae_referentie"],
                scores["mae"],
            ],
        })

        fig_fout = px.bar(
            foutvergelijking,
            x="Model",
            y="Gemiddelde fout",
            labels={"Gemiddelde fout": "Gemiddelde fout (€ per m²)"},
        )

        st.plotly_chart(fig_fout, use_container_width=True)

    st.subheader("Controle van de ondergrens")

    vergelijkingen = []

    for grens, selectie in [(5, model5), (10, model10)]:
        score = bereken_scores(selectie)
        vergelijkingen.append({
            "Minimaal advertenties": grens,
            "Aantal buurten": len(selectie),
            "R²": score["r2"],
            "Gemiddelde fout €/m²": score["mae"],
            "Lagere fout dan referentie (%)": score["verbetering"],
        })

    st.dataframe(
        pd.DataFrame(vergelijkingen).round(2),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "De selecties bevatten verschillende buurten. "
        "De scores zijn daarom niet rechtstreeks vergelijkbaar."
    )

    st.write(
        "**Meetjaarcontrole:** alle kenmerken uit 2024 gaven "
        "bij twintig kruisvalidatieverdelingen telkens een "
        "lagere gemiddelde fout dan de combinatie van recentste "
        "meetjaren. Daarom gebruiken we de gezamenlijke jaargang 2024."
    )


# ============================================================
# 6. TABBLAD: DATA EN KEUZES
# ============================================================

with tab_data:
    st.subheader("Bronnen en meetjaren")

    bronnen = pd.DataFrame({
        "Gegevens": [
            "Vraagprijs en woonoppervlak",
            "Buurtgrenzen",
            "Mediaan besteedbaar huishoudinkomen",
            "Bevolkingsdichtheid",
            "Percentage koopwoningen",
        ],
        "Bron": [
            "Aangeleverd Funda-bestand",
            "Gemeente Amsterdam, Gebieden-API",
            "Gemeente Amsterdam, BBGA",
            "Gemeente Amsterdam, BBGA",
            "Gemeente Amsterdam, BBGA",
        ],
        "Meetmoment": [
            "Oktober 2026, volgens bestandsnaam",
            "Geldig op 3 oktober 2026",
            "2024",
            "1 januari 2024",
            "1 januari 2024",
        ],
    })

    st.dataframe(
        bronnen,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown(
        "[Buurtgrenzen: documentatie]"
        "(https://api.data.amsterdam.nl/v1/docs/datasets/gebieden.html)"
        " · "
        "[BBGA: documentatie]"
        "(https://api.data.amsterdam.nl/v1/docs/datasets/bbga.html)"
    )

    st.subheader("Van advertenties naar analyse")

    selectieoverzicht = pd.DataFrame({
        "Stap": [
            "Alle advertenties",
            "Mogelijke koopwoningen",
            "Met berekenbare prijs per m²",
            "Ook gekoppeld aan een buurt",
            "Buurten met complete kenmerken",
            f"Buurten met minimaal {minimum} advertenties",
        ],
        "Aantal": [
            6864,
            6726,
            6687,
            6674,
            375,
            len(data),
        ],
        "Eenheid": [
            "advertenties",
            "advertenties",
            "advertenties",
            "advertenties",
            "buurten",
            "buurten",
        ],
    })

    st.dataframe(
        selectieoverzicht,
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("Opschoning en ontbrekende gegevens"):
        st.markdown("""
- Parkeerplaatsen en bouwgrond vallen buiten de woninganalyse.
- Eén ontbrekende vraagprijs is aangevuld vanuit `Price`.
- 39 woningen hebben geen berekenbare prijs per m².
- 13 Dialogue-advertenties hebben geen bevestigde buurtcode.
  De punten vallen buiten de opgehaalde buurtvlakken.
- Bij 15 buurten ontbreekt inkomen; bij 1 buurt ontbreekt het percentage koopwoningen. Bevolkingsdichtheid is overal beschikbaar. Ontbrekende waarden zijn niet als nul ingevuld.
- De aantallen ontbrekende waarden wijken af van de casusbeschrijving. In de onderzochte prijs- en woonoppervlaktevelden zijn de beschreven tekstverschuivingen niet aangetroffen.
""")

    with st.expander("Wat kunnen we hieruit niet concluderen?"):
        st.markdown("""
- Vraagprijzen zijn geen gerealiseerde verkoopprijzen.
- Het aangeboden woningpakket vertegenwoordigt niet de hele buurt.
- Advertenties binnen één gebouw of project kunnen sterk samenhangen.
- De buurtkenmerken komen uit 2024, terwijl het woningaanbod uit oktober 2026 komt. Vooral snel veranderende buurten kunnen daardoor minder goed worden beschreven.
- Woningkwaliteit, erfpacht, uitzicht en andere kenmerken ontbreken in het model.
- Een afwijking toont geen causaal effect of bewezen overwaardering.
- Eén momentopname bewijst niet dat een afwijking door de tijd aanhoudt.
""")

    st.subheader("Buurtresultaten downloaden")

    exportkolommen = [
        "buurtcode_schoon",
        "buurt",
        "aantal_woningen",
        "mediaan_prijs_m2",
        "voorspeld_m2_cv",
        "afwijking_m2_cv",
        *KENMERKEN,
        "aandeel_nieuwbouw_pct",
    ]

    export = data[exportkolommen].copy()
    export["jaar_inkomen"] = 2024
    export["jaar_dichtheid"] = 2024
    export["jaar_percentage_koop"] = 2024

    st.download_button(
        label="Download de geselecteerde buurtresultaten",
        data=export.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"buurtresultaten_minimaal_{minimum}.csv",
        mime="text/csv",
    )

    st.caption(
        "Mogelijke dubbele projectadvertenties zijn nog niet "
        "definitief vastgesteld. Vergelijkbare woningen met "
        "verschillende bouwnummers blijven behouden."
    )

