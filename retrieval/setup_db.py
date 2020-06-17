import cv2
import json

paintings = ['Giove ed Europa,Robusti Jacopo detto Tintoretto (Venezia  1518 - 1594),19,000.png',
             'Nerone davanti al corpo di Agrippina,Ferrari Luca detto Luca da Reggio (Reggio Emilia  1599 - Padova  1654),21,001.png',
             'Minerva,Venturini Gaspare (Ferrara  notizie dal 1576 al 1593),20,002.png',
             'La circoncisione di Gesù,Procaccini Giulio Cesare (Bologna  1574 - Milano  1625),22,003.png',
             'Cristo portacroce tra i santi Girolamo e Agostino,Bernardo Parentino o Parenzano (Parenzo  1450 ca. - Mantova ?  post 1496),9,004.png',
             'La Carità romana,Régnier Nicolas detto Nicola Renieri (Maubeuge  1591 – Venezia  1667),21,005.png',
             'Madonna col Bambino in trono fra angeli,Simone di Filippo Benvenuti detto Simone dei Crocifissi (Bologna  1330 ca. - 1399),3,006.png',
             'Sposalizio della Vergine,Schedoni Bartolomeo (Modena  1570 ca. - Parma  1615),20,007.png',
             'Madonna col Bambino tra i santi Giorgio e Michele Arcangelo,Luteri Giovanni di Niccolò detto Dosso Dossi (Tramuschio (Mirandola)  1487 ca. - Ferrara  1542),13,008.png',
             'San Francesco di Paola abbraccia il crocifisso,Stern Ignazio detto Ignazio Stella (Mariahilf  Passavia  1680 ca. - Roma  1748),22,009.png',
             'Cattura di Cristo,Cavedone Giacomo (Sassuolo  1577 - Bologna  1660),21,010.png',
             'Madonna Assunta fra i santi Giovanni Battista e Giorgio,Ferrari Luca detto Luca da Reggio (Reggio Emilia  1599 - Padova  1654),21,011.png',
             'Nascita della Vergine,Scarsella Ippolito detto Scarsellino (Ferrara  1550 ca. - 1620),20,012.png',
             'Ritratti ideali di Ubaldo e Marino d\'Este,Cervi Bernardino (1586 ca. - Modena  1630),20,013.png',
             'Annunciazione; Santa Margherita; Santa Dorotea; Visitazione,,8,014.png',
             'Adorazione dei Magi,Negretti Jacopo detto Palma il giovane (Venezia  1548 ca. - 1628),19,015.png',
             'Presentazione di Gesù al Tempio,Carnevali Domenico (Sassuolo  1524 – Modena  1579),15,016.png',
             'Sant\'Antonio da Padova,Tura Cosmè (Ferrara  1430 ca. - 1495),4,017.png',
             'Compianto sul Cristo morto con i santi Francesco e Bernardino,Cima Giovanni Battista detto Cima da Conegliano (Conegliano  1459 ca. - 1517 ca.),9,018.png',
             'Ritratto d\'uomo,Campi Giulio (attr.) (Cremona  1500 ca. - 1572),16,019.png',
             'Apollo musico,Scarsella Ippolito detto Scarsellino (Ferrara  1550 ca. - 1620),20,020.png',
             'Ritratto del duca Francesco II d\'Este,,16,021.png',
             'I santi Pietro e Paolo,Da Ponte Jacopo detto Bassano (Bassano del Grappa  1510 ca. - 1592),19,022.png',
             'Redentore benedicente,Bastiani Lazzaro (?) (Venezia  1449 ca. - 1512),9,023.png',
             'Il giudizio di Mida,Robusti Jacopo detto Tintoretto (Venezia  1518 - 1594),19,024.png',
             'Ritratti ideali di Matilde di Canossa e Aldobrandino d\'Este,Cervi Bernardino (1586 ca. - Modena  1630),20,025.png',
             'Madonna col Bambino e i santi Sebastiano e Giorgio (?),Luteri Giovanni di Niccolò detto Dosso Dossi (Tramuschio (Mirandola)  1487 ca. - Ferrara  1542),13,026.png',
             'Ritratto di giovane,Giarola Antonio (Verona  1595 - 1665),16,027.png',
             'Madonna in trono col Bambino angeli musicanti e i santi Giovanni Battista Contardo d\'Este e Lucia,Tisi Benvenuto detto Garofalo (Ferrara  1481 ca. - 1559),15,028.png',
             'Madonna col Bambino  santi e donatori,Catena Vincenzo (attr.) (Venezia  1480 ca. - 1531),9,029.png',
             'Ritratti ideali di Alforisio e Acarino d\'Este,Cervi Bernardino (1586 ca. - Modena  1630),20,030.png',
             'Flora,Cignani Carlo (Bologna  1628 – Forlì  1719),22,031.png',
             'Madonna col Bambino e san Giovannino,Cianfanini Giovanni (attr.) (Attivo fra il 1462 e il 1542 ca. a Firenze),8,032.png',
             'Tancredi battezza Clorinda,Badalocchio Sisto (Parma  1585 - post 1620),21,033.png',
             'Plutone,Carracci Agostino (Bologna  1557 - Parma  1602),20,034.png',
             'Ritratto di Quaranta Malvasia,Sabatini Lorenzo (Bologna  1530 ca. - Roma  1576),16,035.png',
             'La buona ventura,Spada Leonello (Bologna  1576 - Parma  1622),21,036.png',
             'Venere e Amore,Carracci Annibale (Bologna  1560 - Roma  1609),20,037.png',
             'Strage dei figli di Niobe,Robusti Jacopo detto Tintoretto (Venezia  1518 - 1594),19,038.png',
             'Madonna col Bambino e santi; Crocifissione; Annunciazione,Agocchiari Barnaba detto Barnaba da Modena  (Doc. a Genova 1361 ca. – 1383 ca.),3,039.png',
             'Ritratto del duca Alfonso IV d\'Este,Suttermans Justus (Anversa  1597 - Firenze  1681),16,040.png',
             'Adorazione dei pastori,Boccaccino Boccaccio (Ferrara  1466/67 – Cremona  1524/25),9,041.png',
             'Madonna col Bambino,Montagna Bartolomeo (Orzinuovi  1449 ca. - Vicenza  1523),9,042.png',
             'Martirio di san Pietro martire,Ferrari Luca detto Luca da Reggio (Reggio Emilia  1599 - Padova  1654),21,043.png',
             'La Regina Tomiri fa immergere nel sangue la testa di Ciro,Ferrari Luca detto Luca da Reggio (Reggio Emilia  1599 - Padova  1654),21,044.png',
             'Ritratto di Maria Beatrice d\'Este Stuart (Maria di Modena) con un paggio,,16,045.png',
             'Gesù crocifisso,Reni Guido (Bologna  1575 - 1642),20,046.png',
             'La Conversazione (Figure allegoriche),Luteri Giovanni di Niccolò detto Dosso Dossi (Tramuschio (Mirandola)  1487 ca. - Ferrara  1542),13,047.png',
             'Santa Maria Maddalena,Cavedone Giacomo (Sassuolo  1577 - Bologna  1660),21,048.png',
             'Madonna col Bambino,Paolo di Giovanni Fei (attr.) (Siena  1369 ca. - 1411),3,049.png',
             'Madonna col Bambino in trono fra angeli,Francesco di Neri da Volterra (Volterra  notizie dal 1338 al 1386 ca.),3,050.png',
             'Minerva,Venturini Gaspare (Ferrara  notizie dal 1576 al 1593),20,051.png',
             'La Musica (Figure allegoriche),Luteri Giovanni di Niccolò detto Dosso Dossi (Tramuschio (Mirandola)  1487 ca. - Ferrara  1542),13,052.png',
             'Ritratto di Maria Farnese d\'Este,Suttermans Justus (attr.) (Anversa  1597 - Firenze  1681),16,053.png',
             'Madonna col Bambino,Panetti Domenico (Ferrara  notizie dal 1489 al 1512 ca.),10,054.png',
             'Madonna col Bambino,Allegri Antonio detto Correggio (Correggio  1489 - 1534),10,055.png',
             'Madonna col Bambino e santi Caterina d\'Alessandria Scolastica Pietro Agostino Giovanni Battista e Paolo,Robusti Jacopo detto Tintoretto (Venezia  1518 - 1594),19,056.png',
             'Nascita del Battista,Liberi Pietro detto Libertino (Padova  1605 – Venezia  1687),19,057.png',
             'L\'Ebbrezza (Figure allegoriche),Luteri Giovanni di Niccolò detto Dosso Dossi (Tramuschio (Mirandola)  1487 ca. - Ferrara  1542),13,058.png',
             'Adorazione del Bambino,Botticini Francesco (Firenze  1446 - 1498),8,059.png',
             'La Seduzione (Figure allegoriche),Luteri Giovanni di Niccolò detto Dosso Dossi (Tramuschio (Mirandola)  1487 ca. - Ferrara  1542),13,060.png',
             'Giuditta con la testa di Oloferne,Bertucci Giacomo detto Jacopone da Faenza (Faenza  1502 - 1579),18,061.png',
             'Giovane imperatore,Venturini Gaspare (Ferrara  notizie dal 1576 al 1593),20,062.png',
             'Ritratto di buffone di corte,Luteri Giovanni di Niccolò detto Dosso Dossi (Tramuschio (Mirandola)  1487 ca. - Ferrara  1542),13,063.png',
             'Apparizione dell\'angelo a Elia,Donducci Giovanni Andrea detto Mastelletta (Bologna  1575 - 1655),20,064.png',
             'Madonna col Bambino,Paolo di Bernardino di Antonio del Signoraccio detto Fra\' Paolino da Pistoia (Pistoia  1488 - 1547),8,065.png',
             'Flora,Carracci Annibale o Carracci Ludovico,20,066.png',
             'L\'Amore (o L\'Abbraccio) (Figure allegoriche),Luteri Giovanni di Niccolò detto Dosso Dossi (Tramuschio (Mirandola)  1487 ca. - Ferrara  1542),13,067.png',
             'Annunciazione,Tisi Benvenuto detto Garofalo (bottega di) (Ferrara  1481 ca. - 1559),15,068.png',
             'Ritratti ideali di Aurelio e Tiberio Azio d\'Este,Cervi Bernardino (1586 ca. - Modena  1630),20,069.png',
             'Vulcano  Minerva e Amore (La nascita di Erittonio),Robusti Jacopo detto Tintoretto (Venezia  1518 - 1594),19,070.png',
             'La dea Latona trasforma i contadini della Licia in rane,Robusti Jacopo detto Tintoretto (Venezia  1518 - 1594),19,071.png',
             'Apparizione dei tre angeli ad Abramo,Donducci Giovanni Andrea detto Mastelletta (Bologna  1575 - 1655),20,072.png',
             'Il Genio dell\'Africa,Venturini Gaspare (Ferrara  notizie dal 1576 al 1593),20,073.png',
             'Madonna della Colonna (Madonna col Bambino),Canozi Cristoforo da Lendinara (Lendinara  1420 ca. - Parma  1490 ca.),4,074.png',
             'Caduta di Fetonte,Robusti Jacopo detto Tintoretto (Venezia  1518 - 1594),19,075.png',
             'Ritratto di fra Giovan Battista da Modena  già duca Alfonso III d\'Este,Loves Matteo (Colonia – Bologna  1647 ca.),16,076.png',
             'Ercole,Belloni Giulio (attr.) (Ferrara  fine XVI secolo – inizi XVII secolo),20,077.png',
             'Imperatore con scudo,Belloni Giulio (attr.) (Ferrara  fine XVI secolo – inizi XVII secolo),20,078.png',
             'Madonna dell\'Umiltà,Serafini Paolo  detto Paolo da Modena (Modena  XIV secolo),3,079.png',
             'Copia dal Ritratto di Ercole I d\'Este,Luteri Giovanni di Niccolò detto Dosso Dossi (da Ercole de\' Roberti) (Tramuschio (Mirandola)  1487 ca. - Ferrara  1542),13,080.png',
             'Ritratto della principessa Anna Maria Maurizia d\'Asburgo,Pourbus Frans II il Giovane (Anversa  1569 - Parigi  1622),16,081.png',
             'Madonna col Bambino; Cristo nel sepolcro; Annunciazione; Santi,Maestro di Torre di Palme (attr.) (Attivo alla fine del XIV secolo a Venezia),3,082.png',
             'Ritratto del duca Francesco I d\'Este,Velázquez Diego Rodríguez de Silva (Siviglia  1599 - Madrid  1660),16,083.png',
             'Priapo insidia Lotide addormentata,Robusti Jacopo detto Tintoretto (Venezia  1518 - 1594),19,084.png',
             'Allegoria della Pace che brucia gli strumenti della guerra,Barbieri Giovan Francesco detto Guercino (Cento  1591 - Bologna  1666),21,085.png',
             'Ritratto di gentiluomo in armatura,Aretusi Cesare (Bologna  1549 - 1612),16,086.png',
             'Re Davide,Cavedone Giacomo (Sassuolo  1577 - Bologna  1660),21,087.png',
             'Piramo e Tisbe,Robusti Jacopo detto Tintoretto (Venezia  1518 - 1594),19,088.png',
             'Ultima cena,Donducci Giovanni Andrea detto Mastelletta (Bologna  1575 - 1655),20,089.png',
             'San Pietro liberato dall\'angelo,Turchi Alessandro detto Orbetto (Verona  1578 - Roma  1649),21,090.png',
             'Allegoria della Pazienza,Filippi Camillo (Ferrara  1500 ca. - 1574)  Filippi Sebastiano detto Bastianino (Ferrara  1528 ca. - 1602),13,091.png',
             'Ecce Homo,Solario Antonio detto Zingaro (Civita d’Antino  1465 ca. - Napoli  1530),9,092.png',
             'La Fama,Scarsella Ippolito detto Scarsellino (Ferrara  1550 ca. - 1620),20,093.png',
             'Galatea,Carracci Ludovico (Bologna  1555 - 1619),20,094.png',
             ]


def main():
    ims = []
    im = {}
    orb = cv2.ORB_create()
    for painting in paintings:
        paint = painting.split(',')
        title = paint[0]
        author = paint[1]
        room = paint[2]
        image = paint[3]
        img = cv2.imread('./paintings_db/' + image, 0)
        kps, dscs = orb.detectAndCompute(img, None)
        im['title'] = title
        im['author'] = author
        im['room'] = room
        im['image'] = image
        im['keypoints'] = [[kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id] for kp in kps]
        dscs = dscs.tolist()
        im['descriptors'] = dscs
        ims.append(im)
        im = {}
    with open('./paintings_descriptors.json', 'w') as f:
        json.dump(ims, f)


if __name__ == "__main__":
    main()
