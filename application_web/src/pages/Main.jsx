import { useState } from 'react'
import { MeshGradient } from '@paper-design/shaders-react'
import Classify from '../components/Classify'
import Stitch from '../components/Stitch'
import Gan from '../components/Gan'
import Yolo from '../components/Yolo'
import FullProcess from '../components/FullProcess'
import './Landing.css'

const API_BASE = '/api'

const baseUrl = "https://github.com/luscious162/oracle_CCCC/releases/download/weight/"

const db = {
  cls: ['bone1.jpg', 'bone2.jpg', 'bone3.jpg', 'bone4.jpg', 'bone5.jpg', 'shell1.jpg', 'shell2.jpg', 'shell3.jpg', 'shell4.jpg', 'shell5.jpg'],
  stitch: [{a: 'rejion1_1.png', b: 'rejion1_2.png'}, {a: 'rejion2_1.png', b: 'rejion2_2.png'}, {a: 'rejion3_1.png', b: 'rejion3_2.png'}],
  gan: ['gan1.jpg', 'gan2.jpg', 'gan3.jpg'],
  yolo: ['yolo1.jpg', 'yolo2.jpg', 'yolo3.jpg']
}

export default function Main() {
  const [activeTab, setActiveTab] = useState('classify')

  const tabs = [
    { id: 'classify', label: '分类识别' },
    { id: 'stitch', label: '碎片缀合' },
    { id: 'gan', label: '摹本生成' },
    { id: 'yolo', label: '甲骨文检测与释义' },
    { id: 'full', label: '全流程研析' }
  ]

  return (
    <div
      className="landing-bg"
      style={{
        minHeight: '100vh',
        position: 'relative',
        padding: '40px 20px'
      }}
    >
      <MeshGradient
        speed={1}
        scale={1}
        distortion={0.8}
        swirl={0.1}
        colors={['#E0D68A', '#CB9173', '#8E443D', '#511730']}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          zIndex: 0
        }}
      />

      <div style={{
        position: 'relative',
        zIndex: 1,
        maxWidth: '1200px',
        margin: '0 auto',
        background: 'rgba(255, 255, 255, 0.92)',
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
        padding: '40px',
        borderRadius: '24px',
        boxShadow: '0 16px 48px rgba(81, 23, 48, 0.2)',
        border: '1px solid rgba(255, 255, 255, 0.5)'
      }}>
        <div style={{
          textAlign: 'center',
          marginBottom: '40px',
          position: 'relative'
        }}>
          <div style={{
            position: 'absolute',
            bottom: '-20px',
            left: '50%',
            transform: 'translateX(-50%)',
            width: '80px',
            height: '4px',
            background: 'linear-gradient(90deg, #8E443D, #511730)',
            borderRadius: '2px'
          }} />
          <h2 style={{
            fontSize: '32px',
            fontWeight: 700,
            background: 'linear-gradient(135deg, #511730 0%, #8E443D 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            letterSpacing: '4px',
            marginBottom: '8px'
          }}>
            甲骨识微
          </h2>
          <p style={{
            color: '#5A5A5A',
            fontSize: '14px',
            letterSpacing: '2px'
          }}>
            ORACLE BONE INSCRIPTION INTELLIGENT ANALYSIS
          </p>
        </div>

        <div style={{
          display: 'flex',
          borderBottom: '2px solid rgba(81, 23, 48, 0.1)',
          marginBottom: '32px',
          flexWrap: 'wrap',
          gap: '8px'
        }}>
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: '14px 28px',
                cursor: 'pointer',
                border: 'none',
                background: activeTab === tab.id ? 'rgba(81, 23, 48, 0.04)' : 'transparent',
                fontSize: '15px',
                fontWeight: 600,
                transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                color: activeTab === tab.id ? '#511730' : '#5A5A5A',
                position: 'relative',
                borderRadius: '8px 8px 0 0'
              }}
            >
              {tab.label}
              {activeTab === tab.id && (
                <div style={{
                  position: 'absolute',
                  bottom: '-2px',
                  left: 0,
                  width: '100%',
                  height: '3px',
                  background: 'linear-gradient(90deg, #8E443D, #511730)',
                  borderRadius: '2px'
                }} />
              )}
            </button>
          ))}
        </div>

        <div style={{ display: activeTab === 'classify' ? 'block' : 'none', animation: 'fadeIn 0.5s' }}>
          <Classify baseUrl={baseUrl} db={db} apiBase={API_BASE} />
        </div>
        <div style={{ display: activeTab === 'stitch' ? 'block' : 'none', animation: 'fadeIn 0.5s' }}>
          <Stitch baseUrl={baseUrl} db={db} apiBase={API_BASE} />
        </div>
        <div style={{ display: activeTab === 'gan' ? 'block' : 'none', animation: 'fadeIn 0.5s' }}>
          <Gan baseUrl={baseUrl} db={db} apiBase={API_BASE} />
        </div>
        <div style={{ display: activeTab === 'yolo' ? 'block' : 'none', animation: 'fadeIn 0.5s' }}>
          <Yolo baseUrl={baseUrl} db={db} apiBase={API_BASE} />
        </div>
        <div style={{ display: activeTab === 'full' ? 'block' : 'none', animation: 'fadeIn 0.5s' }}>
          <FullProcess baseUrl={baseUrl} db={db} apiBase={API_BASE} isActive={activeTab === 'full'} />
        </div>
      </div>
    </div>
  )
}
