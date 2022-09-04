#pragma once
#include <cinttypes>
#include <memory>
#include <vector>
#include <cmath>
#include <spdlog/spdlog.h>



struct measurement
{
    int m_engine_id;
    
    double m_setting1;
    double m_setting2;
    double m_temp_lpc_outlet;
    double m_temp_hpc_outlet;
    double m_temp_lpt_outlet;
    double m_pressure_hpc_outlet;
    double m_physical_fan_speed;
    double m_physical_core_speed;
    double m_static_pressure_hpc_outlet;
    double m_fuel_flow_ration_ps30;
    double m_corrected_fan_speed;
    double m_corrected_core_speed;
    double m_bypass_ratio;
    double m_bleed_enthalpy;
    double m_hpt_collant_bleed;
    double m_lpt_coolant_bleed;

    measurement() = default;

    measurement(int engine_id, double setting1, double setting2, double temp_lpc_outlet, double temp_hpc_outlet, double temp_lpt_outlet, double pressure_hpc_outlet, double physical_fan_speed, double physical_core_speed, double static_pressure_hpc_outlet, double fuel_flow_ration_ps30, double corrected_fan_speed, double corrected_core_speed, double bypass_ratio, double bleed_enthalpy, double hpt_collant_bleed, double lpt_coolant_bleed)
    {
        m_engine_id = engine_id;
        
        m_setting1 = setting1;
        m_setting2 = setting2;
        m_temp_lpc_outlet = temp_lpc_outlet;
        m_temp_hpc_outlet = temp_hpc_outlet;
        m_temp_lpt_outlet = temp_lpt_outlet;
        m_pressure_hpc_outlet = pressure_hpc_outlet;
        m_physical_fan_speed = physical_fan_speed;
        m_physical_core_speed = physical_core_speed;
        m_static_pressure_hpc_outlet = static_pressure_hpc_outlet;
        m_fuel_flow_ration_ps30 = fuel_flow_ration_ps30;
        m_corrected_fan_speed = corrected_fan_speed;
        m_corrected_core_speed = corrected_core_speed;
        m_bypass_ratio = bypass_ratio;
        m_bleed_enthalpy = bleed_enthalpy;
        m_hpt_collant_bleed = hpt_collant_bleed;
        m_lpt_coolant_bleed = lpt_coolant_bleed;
    }

    void sum(measurement* other){
        m_engine_id += other->m_engine_id;
        
        m_setting1 += other->m_setting1;
        m_setting2 += other->m_setting2;
        m_temp_lpc_outlet += other->m_temp_lpc_outlet;
        m_temp_hpc_outlet += other->m_temp_hpc_outlet;
        m_temp_lpt_outlet += other->m_temp_lpt_outlet;
        m_pressure_hpc_outlet += other->m_pressure_hpc_outlet;
        m_physical_fan_speed += other->m_physical_fan_speed;
        m_physical_core_speed += other->m_physical_core_speed;
        m_static_pressure_hpc_outlet += other->m_static_pressure_hpc_outlet;
        m_fuel_flow_ration_ps30 += other->m_fuel_flow_ration_ps30;
        m_corrected_fan_speed += other->m_corrected_fan_speed;
        m_corrected_core_speed += other->m_corrected_core_speed;
        m_bypass_ratio += other->m_bypass_ratio;
        m_bleed_enthalpy += other->m_bleed_enthalpy;
        m_hpt_collant_bleed += other->m_hpt_collant_bleed;
        m_lpt_coolant_bleed += other->m_lpt_coolant_bleed;
    }

    std::unique_ptr<measurement> divide(double divider)
    {
        double engine_id = m_engine_id / divider;
        
        double setting1 = m_setting1 / divider;
        double setting2 = m_setting2 / divider;
        double temp_lpc_outlet = m_temp_lpc_outlet / divider;
        double temp_hpc_outlet = m_temp_hpc_outlet / divider;
        double temp_lpt_outlet = m_temp_lpt_outlet / divider;
        double pressure_hpc_outlet = m_pressure_hpc_outlet / divider;
        double physical_fan_speed = m_physical_fan_speed / divider;
        double physical_core_speed = m_physical_core_speed / divider;
        double static_pressure_hpc_outlet = m_static_pressure_hpc_outlet / divider;
        double fuel_flow_ration_ps30 = m_fuel_flow_ration_ps30 / divider;
        double corrected_fan_speed = m_corrected_fan_speed / divider;
        double corrected_core_speed = m_corrected_core_speed / divider;
        double bypass_ratio = m_bypass_ratio / divider;
        double bleed_enthalpy = m_bleed_enthalpy / divider;
        double hpt_collant_bleed = m_hpt_collant_bleed / divider;
        double lpt_coolant_bleed = m_lpt_coolant_bleed / divider;

        std::unique_ptr<measurement> divided = std::make_unique<measurement>((int)engine_id, setting1, setting2, temp_lpc_outlet, temp_hpc_outlet, temp_lpt_outlet, pressure_hpc_outlet, physical_fan_speed, physical_core_speed, static_pressure_hpc_outlet, fuel_flow_ration_ps30, corrected_fan_speed, corrected_core_speed, bypass_ratio, bleed_enthalpy, hpt_collant_bleed, lpt_coolant_bleed);

        return divided;
    }

    std::unique_ptr<measurement> std_dev(std::vector<std::unique_ptr<measurement>> all_measurements, std::unique_ptr<measurement>* means, double count){
        double engine_id, setting1, setting2, temp_lpc_outlet,temp_hpc_outlet, temp_lpt_outlet, pressure_hpc_outlet, physical_fan_speed, physical_core_speed ,static_pressure_hpc_outlet, fuel_flow_ration_ps30, corrected_fan_speed, corrected_core_speed, bypass_ratio, bleed_enthalpy, hpt_collant_bleed, lpt_coolant_bleed;
        
        for (size_t i = 0; i < all_measurements.size(); i++) 
        {
            spdlog::info("first iteration");
            spdlog::info("engine id: {}",  all_measurements.at(i)->m_engine_id);
            // int test = all_measurements.at(i)->m_engine_id - means.m_engine_id; // segmentation fault
            // spdlog::info("test: {}", test);

            // engine_id = std::sqrt ( std::pow(all_measurements.at(i).m_engine_id - means.m_engine_id, 2) / count );
            // setting1 = std::sqrt ( std::pow(all_measurements.at(i).m_setting1 - means.m_setting1, 2) / count ) ;
            // setting2 = std::sqrt ( std::pow(all_measurements.at(i).m_setting2 - means.m_setting2, 2) / count ) ;
            // temp_lpc_outlet = std::sqrt ( std::pow(all_measurements.at(i).m_temp_lpc_outlet - means.m_temp_lpc_outlet, 2) / count ) ;
            // temp_hpc_outlet = std::sqrt ( std::pow(all_measurements.at(i).m_temp_hpc_outlet - means.m_temp_hpc_outlet, 2) / count ) ;
            // temp_lpt_outlet = std::sqrt ( std::pow(all_measurements.at(i).m_temp_lpt_outlet - means.m_temp_lpt_outlet, 2) / count ) ;
            // pressure_hpc_outlet = std::sqrt ( std::pow(all_measurements.at(i).m_pressure_hpc_outlet - means.m_pressure_hpc_outlet, 2) / count ) ;
            // physical_fan_speed = std::sqrt ( std::pow(all_measurements.at(i).m_physical_fan_speed - means.m_physical_fan_speed, 2) / count ) ;
            // physical_core_speed = std::sqrt ( std::pow(all_measurements.at(i).m_physical_core_speed - means.m_physical_core_speed, 2) / count ) ;
            // static_pressure_hpc_outlet = std::sqrt ( std::pow(all_measurements.at(i).m_static_pressure_hpc_outlet - means.m_static_pressure_hpc_outlet, 2) / count ) ;
            // fuel_flow_ration_ps30 = std::sqrt ( std::pow(all_measurements.at(i).m_fuel_flow_ration_ps30 - means.m_fuel_flow_ration_ps30, 2) / count ) ;
            // corrected_fan_speed = std::sqrt ( std::pow(all_measurements.at(i).m_corrected_fan_speed - means.m_corrected_fan_speed, 2) / count ) ;
            // corrected_core_speed = std::sqrt ( std::pow(all_measurements.at(i).m_corrected_core_speed - means.m_corrected_core_speed, 2) / count ) ;
            // bypass_ratio = std::sqrt ( std::pow(all_measurements.at(i).m_bypass_ratio - means.m_bypass_ratio, 2) / count ) ;
            // bleed_enthalpy = std::sqrt ( std::pow(all_measurements.at(i).m_bleed_enthalpy - means.m_bleed_enthalpy, 2) / count ) ;
            // hpt_collant_bleed = std::sqrt ( std::pow(all_measurements.at(i).m_hpt_collant_bleed - means.m_hpt_collant_bleed, 2) / count ) ;
            // lpt_coolant_bleed = std::sqrt ( std::pow(all_measurements.at(i).m_lpt_coolant_bleed - means.m_lpt_coolant_bleed, 2) / count ) ;
        }

        spdlog::info("creating new measurement");
        
        std::unique_ptr<measurement> std_dev = std::make_unique<measurement>((int)engine_id, setting1, setting2, temp_lpc_outlet, temp_hpc_outlet, temp_lpt_outlet, pressure_hpc_outlet, physical_fan_speed, physical_core_speed, static_pressure_hpc_outlet, fuel_flow_ration_ps30, corrected_fan_speed, corrected_core_speed, bypass_ratio, bleed_enthalpy, hpt_collant_bleed, lpt_coolant_bleed);

        return std_dev;
    }
};