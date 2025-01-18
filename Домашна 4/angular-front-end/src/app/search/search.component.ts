import { Component, ViewChild, ElementRef} from '@angular/core';
import {RouterLink} from "@angular/router";
import { Router } from "@angular/router";

@Component({
  selector: 'app-search',
  standalone: true,
  imports: [
    RouterLink
  ],
  templateUrl: './search.component.html',
  styleUrl: './search.component.scss'
})
export class SearchComponent {

  constructor(private router: Router) {}

  ngOnInit(){
    const words = [
        'ADIN', 'ALK', 'ALKB', 'AMBR', 'AMEH', 'APTK', 'ATPP', 'AUMK', 'BANA', 'BGOR', 'BIKF', 'BIM', 'BLTU', 'CBNG', 'CDHV', 'CEVI', 'CKB', 'CKBKO', 'DEBA', 'DIMI', 'EDST', 'ELMA', 'ELNC', 'ENER', 'ENSA', 'EUHA', 'EUMK', 'EVRO', 'FAKM', 'FERS', 'FKTL', 'FROT', 'FUBT', 'GALE', 'GDKM', 'GECK', 'GECT', 'GIMS', 'GRDN', 'GRNT', 'GRSN', 'GRZD', 'GTC', 'GTRG', 'IJUG', 'INB', 'INHO', 'INOV', 'INPR', 'INTP', 'JAKO', 'JUSK', 'KARO', 'KDFO', 'KJUBI', 'KKST', 'KLST', 'KMB', 'KMPR', 'KOMU', 'KONF', 'KONZ', 'KORZ', 'KPSS', 'KULT', 'KVAS', 'LAJO', 'LHND', 'LOTO', 'LOZP', 'MAGP', 'MAKP', 'MAKS', 'MB', 'MERM', 'MKSD', 'MLKR', 'MODA', 'MPOL', 'MPT', 'MPTE', 'MTUR', 'MZHE', 'MZPU', 'NEME', 'NOSK', 'OBPP', 'OILK', 'OKTA', 'OMOS', 'OPFO', 'OPTK', 'ORAN', 'OSPO', 'OTEK', 'PELK', 'PGGV', 'PKB', 'POPK', 'PPIV', 'PROD', 'PROT', 'PTRS', 'RADE', 'REPL', 'RIMI', 'RINS', 'RZEK', 'RZIT', 'RZIZ', 'RZLE', 'RZLV', 'RZTK', 'RZUG', 'RZUS', 'SBT', 'SDOM', 'SIL', 'SKON', 'SKP', 'SLAV', 'SNBT', 'SNBTO', 'SOLN', 'SPAZ', 'SPAZP', 'SPOL', 'SSPR', 'STB', 'STBP', 'STIL', 'STOK', 'TAJM', 'TBKO', 'TEAL', 'TEHN', 'TEL', 'TETE', 'TIKV', 'TKPR', 'TKVS', 'TNB', 'TRDB', 'TRPS', 'TRUB', 'TSMP', 'TSZS', 'TTK', 'TTKO', 'UNI', 'USJE', 'VARG', 'VFPM', 'VITA', 'VROS', 'VSC', 'VTKS', 'ZAS', 'ZILU', 'ZILUP', 'ZIMS', 'ZKAR', 'ZPKO', 'ZPOG', 'ZUAS']
    const containerEl = document.querySelector('.container')
    const formEl = document.querySelector('#search')
    const dropEl = document.querySelector('.drop')

    // @ts-ignore
    const formHandler = (e) => {
      const userInput = e.target.value.toLowerCase()

      if(userInput.length === 0) {
        // @ts-ignore
        dropEl.style.height = 0
        // @ts-ignore
        return dropEl.innerHTML = ''
      }

      const filteredWords = words.filter(word => word.toLowerCase().includes(userInput)).sort().splice(0, 5)

      // @ts-ignore
      dropEl.innerHTML = ''
      filteredWords.forEach(item => {
        const listEl = document.createElement('li')
        listEl.textContent = item
        if(item === userInput) {
          listEl.classList.add('match')
        }
        listEl.addEventListener('click', function() {
          // @ts-ignore
          document.getElementById('search').value = item
        });
        // @ts-ignore
        dropEl.appendChild(listEl)
      })

      // @ts-ignore
      if(dropEl.children[0] === undefined) {
        // @ts-ignore
        return dropEl.style.height = 0
      }
      // @ts-ignore
      let totalChildrenHeight = dropEl.children[0].offsetHeight * filteredWords.length
      // @ts-ignore
      dropEl.style.height = totalChildrenHeight + 'px'

    }
    // @ts-ignore
    formEl.addEventListener('input', formHandler)
  }


  getStockData(symbol: string) {
    this.router.navigate(['/stocks'], { queryParams: { symbol: symbol } });
    // window.location.href = '/stocks?symbol=' + symbol;
  }
}
