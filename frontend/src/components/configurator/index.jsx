import { useEffect, useRef, useState } from "react"
import styles from "./configurator.module.css"
import { max } from "lodash"

function GridPoint({active, i, j, setGrid}){
    function onClickHandler(i,j){
        setGrid(max([i,j]))
    }

    return <p className={`${i <= active && j <= active ? styles.active : styles.inactive} ${styles.gridPoint}` } 
              onClick={() => onClickHandler(i, j)}></p>
}

function Grid(){
    const divRef = useRef(null)
    const [temp, setTemp] = useState([])
    const [cols, setCols] = useState(4)
    
    function setGridHelper(point){
        setCols(point)
    }

    function makeGrid(){
        const count = 5
        const children = []
        for (let i = 1; i <= count; i++){
            for (let j = 1; j <= count; j++){
                children.push(<GridPoint active={cols} i={i} j={j} setGrid={setGridHelper} />)
            }
        }
        setTemp(children)
    }

    useEffect(() => {
        makeGrid()
    }, [cols])

    return <div className={styles.gridParent} ref={divRef}>
        {temp.map((child, index) => <span key={index}>{child}</span>)}
    </div>
}

export default function Configurator(){
    const [numImages, setNumImages] = useState(2)

    return <div style={{marginLeft: 20, paddingLeft: 20, borderLeft: '1px solid #555'}}>
        Configurator
        <Grid />

        <div className={styles.count_input}>
        <label >Number of Images</label>
        <input type="number"  value={numImages} onChange={(e) => setNumImages(e.target.value)} />
        </div>
    </div>
}